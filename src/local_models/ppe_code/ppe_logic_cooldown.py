import time
import cv2
from collections import defaultdict, deque


class PPELogic:
    def __init__(
        self,
        model_path=None,
        grace_frames=20,
        confirm_frames=5,
        resolve_frames=3,
        state_ttl_frames=60,
        emit_violation_updates=False,
        anon_iou_threshold=0.35,
        alert_cooldown_seconds=900,
        equipment_hold_frames=90,
    ):
        # Thresholds per class
        self.class_thresholds = {
            0: 0.4,   # boots
            1: 0.5,   # helmet
            2: 0.3,   # no boots
            3: 0.35,  # no helmet
            4: 0.25,  # no vest
            5: 0.5,   # person
            6: 0.65,  # vest
        }

        # Colors for drawing
        self.class_colors = {
            0: (255, 0, 0),
            1: (0, 255, 255),
            2: (0, 0, 255),
            3: (255, 0, 255),
            4: (0, 255, 0),
            5: (0, 165, 255),
            6: (128, 0, 128),
        }

        # Rolling average score buffer per person and class
        self.score_buffers = defaultdict(lambda: defaultdict(lambda: deque(maxlen=30)))

        # Temporal alert tuning
        self.grace_frames = int(grace_frames)
        self.confirm_frames = int(confirm_frames)
        self.resolve_frames = int(resolve_frames)
        self.state_ttl_frames = int(state_ttl_frames)
        self.emit_violation_updates = bool(emit_violation_updates)
        self.anon_iou_threshold = float(anon_iou_threshold)
        self.alert_cooldown_seconds = int(alert_cooldown_seconds)
        self.equipment_hold_frames = int(equipment_hold_frames)

        # Track per-person alert lifecycle
        self.person_states = defaultdict(
            lambda: {
                "pending": False,
                "pending_since": None,
                "bad_streak": 0,
                "good_streak": 0,
                "alert_active": False,
                "last_missing": set(),
                "last_seen_frame": -1,
                "last_yes_frame": {
                    "boots": -10**9,
                    "helmet": -10**9,
                    "vest": -10**9,
                },
            }
        )

        # Fallback tracking for frames where detector returns pid=-1.
        self._anon_next_id = 1
        self._anon_tracks = {}

        # Duplicate alert suppression memory
        # key: (person_key, status, missing_signature) -> monotonic timestamp
        self._last_emitted_alert_time = {}

    def _frame_index(self, frame_num):
        if isinstance(frame_num, int) and frame_num > 0:
            return frame_num

        if not hasattr(self, "_internal_frame_index"):
            self._internal_frame_index = 0

        self._internal_frame_index += 1
        return self._internal_frame_index

    def _cleanup_stale_states(self, frame_idx):
        stale_ids = []
        for pid, state in self.person_states.items():
            if frame_idx - state["last_seen_frame"] > self.state_ttl_frames:
                stale_ids.append(pid)

        for pid in stale_ids:
            self.person_states.pop(pid, None)
            self.score_buffers.pop(pid, None)

        stale_anon = []
        for anon_id, track in self._anon_tracks.items():
            if frame_idx - track["last_seen_frame"] > self.state_ttl_frames:
                stale_anon.append(anon_id)
        for anon_id in stale_anon:
            self._anon_tracks.pop(anon_id, None)

    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0, inter_x2 - inter_x1)
        ih = max(0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    def _assign_fallback_id(self, bbox, frame_idx):
        best_id = None
        best_iou = 0.0
        for anon_id, track in self._anon_tracks.items():
            iou = self._bbox_iou(bbox, track["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_id = anon_id

        if best_id is not None and best_iou >= self.anon_iou_threshold:
            self._anon_tracks[best_id]["bbox"] = bbox
            self._anon_tracks[best_id]["last_seen_frame"] = frame_idx
            return f"anon_{best_id}"

        new_id = self._anon_next_id
        self._anon_next_id += 1
        self._anon_tracks[new_id] = {"bbox": bbox, "last_seen_frame": frame_idx}
        return f"anon_{new_id}"

    @staticmethod
    def _missing_signature(missing_set):
        if not missing_set:
            return "none"
        return "|".join(sorted(missing_set))

    def _emit_with_cooldown(self, alerts, person_key, status, comparisons, bbox, frame_idx, missing_set):
        sig = (str(person_key), status, self._missing_signature(missing_set))
        now = time.monotonic()
        last = self._last_emitted_alert_time.get(sig)

        if last is not None and (now - last) < self.alert_cooldown_seconds:
            return

        self._last_emitted_alert_time[sig] = now
        alerts.append(
            {
                "person_id": person_key,
                "status": status,
                "ppe_status": comparisons,
                "bbox": bbox,
                "frame_num": frame_idx,
            }
        )

    def process_frame(self, result, frame_num=1):
        frame_idx = self._frame_index(frame_num)
        frame = result.orig_img.copy()

        detections_json = []
        alerts = []

        persons = []
        others = []

        # Person detection
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            if conf < self.class_thresholds.get(cls_id, 0.5):
                continue

            if cls_id == 5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pid = int(box.id.item()) if box.id is not None else -1
                persons.append((pid, (x1, y1, x2, y2)))

                label = f"ID:{pid} person {conf:.2f}"
                color = self.class_colors.get(cls_id, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # PPE detection
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            if cls_id == 5 or conf < self.class_thresholds.get(cls_id, 0.5):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            others.append((cls_id, (x1, y1, x2, y2)))

            label = f"{result.names[cls_id]} {conf:.2f}"
            color = self.class_colors.get(cls_id, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # PPE logic per person
        for _, (pid, (px1, py1, px2, py2)) in enumerate(persons):
            state_pid = pid if pid != -1 else self._assign_fallback_id((px1, py1, px2, py2), frame_idx)

            for cls_id, (x1, y1, x2, y2) in others:
                inside = x1 > px1 and y1 > py1 and x2 < px2 and y2 < py2
                score = 1.0 if inside else 0.0
                self.score_buffers[state_pid][cls_id].append(score)

            avg_scores = {}
            for cid, buf in self.score_buffers[state_pid].items():
                avg_scores[result.names[cid]] = sum(buf) / len(buf)

            avg_scores["person"] = 1.0

            comparisons = {
                "boots": "yes" if avg_scores.get("boots", 0) > avg_scores.get("no boots", 0) else "no",
                "helmet": "yes" if avg_scores.get("helmet", 0) > avg_scores.get("no helmet", 0) else "no",
                "vest": "yes" if avg_scores.get("vest", 0) > avg_scores.get("no vest", 0) else "no",
            }

            state = self.person_states[state_pid]
            state["last_seen_frame"] = frame_idx

            # Hold positive PPE evidence for a short window to avoid false alerts
            # from temporary misses/occlusion (e.g., vest not detected for a few frames).
            effective = dict(comparisons)
            for item in ("boots", "helmet", "vest"):
                if comparisons[item] == "yes":
                    state["last_yes_frame"][item] = frame_idx
                elif frame_idx - state["last_yes_frame"][item] <= self.equipment_hold_frames:
                    effective[item] = "yes"

            detections_json.append(
                {
                    "person_id": state_pid,
                    "avg_scores": avg_scores,
                    "ppe_status": effective,
                    "raw_ppe_status": comparisons,
                    "bbox": [px1, py1, px2, py2],
                }
            )

            summary = f"H:{effective['helmet']} V:{effective['vest']} B:{effective['boots']}"
            cv2.putText(frame, summary, (px1, py2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            missing = {k for k, v in effective.items() if v == "no"}
            is_safe = len(missing) == 0
            bbox = [px1, py1, px2, py2]

            if is_safe:
                state["bad_streak"] = 0
                state["good_streak"] += 1

                if state["pending"] and not state["alert_active"] and state["good_streak"] >= self.resolve_frames:
                    state["pending"] = False
                    state["pending_since"] = None

                if state["alert_active"] and state["good_streak"] >= self.resolve_frames:
                    self._emit_with_cooldown(
                        alerts,
                        state_pid,
                        "violation_resolved",
                        effective,
                        bbox,
                        frame_idx,
                        set(),
                    )
                    state["alert_active"] = False
                    state["pending"] = False
                    state["pending_since"] = None
                    state["last_missing"] = set()

            else:
                state["good_streak"] = 0
                state["bad_streak"] += 1

                if not state["alert_active"]:
                    if not state["pending"]:
                        state["pending"] = True
                        state["pending_since"] = frame_idx

                    elapsed = frame_idx - state["pending_since"]
                    if elapsed >= self.grace_frames and state["bad_streak"] >= self.confirm_frames:
                        self._emit_with_cooldown(
                            alerts,
                            state_pid,
                            "violation_started",
                            effective,
                            bbox,
                            frame_idx,
                            missing,
                        )
                        state["alert_active"] = True
                        state["pending"] = False
                        state["pending_since"] = None
                        state["last_missing"] = set(missing)

                else:
                    if self.emit_violation_updates and missing != state["last_missing"]:
                        self._emit_with_cooldown(
                            alerts,
                            state_pid,
                            "violation_updated",
                            effective,
                            bbox,
                            frame_idx,
                            missing,
                        )
                        state["last_missing"] = set(missing)

        self._cleanup_stale_states(frame_idx)
        return frame, detections_json, alerts if alerts else None
