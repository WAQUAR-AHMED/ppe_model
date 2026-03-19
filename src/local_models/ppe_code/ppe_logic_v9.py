"""
ppe_logic_v9.py - Pose-gated PPE logic for the 7-class best_new.pt model.
"""

import math
import time
from collections import defaultdict, deque

import cv2


def _normalise(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _canonical_item_name(norm: str) -> str:
    if "hardhat" in norm or "helmet" in norm:
        return "helmet"
    if "mask" in norm:
        return "mask"
    if "glass" in norm or "goggle" in norm:
        return "glasses"
    if "vest" in norm:
        return "safety_vest"
    if norm == "boot" or "boots" in norm:
        return "boots"
    if "glove" in norm:
        return "glove"
    return norm


def _build_class_maps(class_names: dict):
    person_id = None
    ppe_classes = {}
    for cid, raw_name in class_names.items():
        norm = _normalise(raw_name)
        if norm == "person":
            person_id = cid
        else:
            ppe_classes[cid] = _canonical_item_name(norm)
    return person_id, ppe_classes


_DEFAULT_COLORS = [
    (255, 255, 255),
]
_PERSON_COLOR = (0, 165, 255)
_REGION_COLORS = {
    "head": (0, 200, 255),
    "torso": (0, 255, 200),
    "feet": (255, 200, 0),
    "hands": (255, 120, 120),
}
_ESSENTIAL_PPE = {"helmet", "safety_vest", "boots"}
_NON_ESSENTIAL_PPE = {"mask", "glasses", "glove"}


class PPELogicV9:
    def __init__(
        self,
        model_class_names: dict,
        required_ppe=None,
        show_regions=False,
        class_conf_thresholds=None,
        grace_frames=90,
        confirm_frames=5,
        resolve_frames=3,
        state_ttl_frames=120,
        emit_violation_updates=False,
        alert_cooldown_seconds=900,
        equipment_hold_frames=90,
        reid_max_gap_frames=30,
        reid_min_iou=0.30,
        reid_max_center_distance=60.0,
        overlap_threshold=0.0,
    ):
        self.person_class_id, self.ppe_class_map = _build_class_maps(model_class_names)
        if self.person_class_id is None:
            raise ValueError(
                "[PPELogicV9] Could not find a 'person' class in model class names: "
                + str(model_class_names)
            )

        self.class_colors = {}
        for i, cid in enumerate(sorted(self.ppe_class_map.keys())):
            self.class_colors[cid] = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        self.class_colors[self.person_class_id] = _PERSON_COLOR

        if required_ppe is not None:
            req_norm = {_canonical_item_name(_normalise(r)) for r in required_ppe}
            self.required_ppe_ids = {
                cid for cid, name in self.ppe_class_map.items() if name in req_norm
            }
        else:
            self.required_ppe_ids = set(self.ppe_class_map.keys())
        self.show_regions = bool(show_regions)

        self.class_thresholds = defaultdict(lambda: 0.4)
        self.class_thresholds[self.person_class_id] = 0.5
        if class_conf_thresholds:
            for class_name, thresh in class_conf_thresholds.items():
                norm_name = _normalise(class_name)
                if norm_name == "person":
                    self.class_thresholds[self.person_class_id] = float(thresh)
                    continue
                wanted = _canonical_item_name(norm_name)
                for cid, item_name in self.ppe_class_map.items():
                    if item_name == wanted:
                        self.class_thresholds[cid] = float(thresh)

        self.grace_frames = int(grace_frames)
        self.confirm_frames = int(confirm_frames)
        self.resolve_frames = int(resolve_frames)
        self.state_ttl_frames = int(state_ttl_frames)
        self.emit_violation_updates = bool(emit_violation_updates)
        self.alert_cooldown_seconds = int(alert_cooldown_seconds)
        self.equipment_hold_frames = int(equipment_hold_frames)
        self.overlap_threshold = float(overlap_threshold)
        self.reid_max_gap_frames = int(reid_max_gap_frames)
        self.reid_min_iou = float(reid_min_iou)
        self.reid_max_center_distance = float(reid_max_center_distance)

        self.score_buffers = defaultdict(lambda: defaultdict(lambda: deque(maxlen=30)))

        def _make_state():
            last_yes = {name: -(10 ** 9) for name in self.ppe_class_map.values()}
            return {
                "pending": False,
                "pending_since": None,
                "bad_streak": 0,
                "good_streak": 0,
                "alert_active": False,
                "last_missing": set(),
                "last_seen_frame": -1,
                "last_yes_frame": last_yes,
            }

        self.person_states = defaultdict(_make_state)
        self._last_emitted_alert_time = {}
        self.raw_to_stable = {}
        self.stable_tracks = {}
        self._next_stable_id = 1
        self._internal_frame_index = 0

    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    @staticmethod
    def _bbox_center(b):
        x1, y1, x2, y2 = b
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _center_distance(self, a, b):
        ax, ay = self._bbox_center(a)
        bx, by = self._bbox_center(b)
        return math.hypot(ax - bx, ay - by)

    def _frame_index(self, frame_num):
        if isinstance(frame_num, int) and frame_num > 0:
            return frame_num
        self._internal_frame_index += 1
        return self._internal_frame_index

    def _new_stable_id(self):
        sid = self._next_stable_id
        self._next_stable_id += 1
        return sid

    def _match_recent_stable(self, bbox, frame_idx, used_stable_ids):
        best_sid, best_iou, best_dist = None, 0.0, float("inf")
        for sid, tr in self.stable_tracks.items():
            if sid in used_stable_ids:
                continue
            if frame_idx - tr["last_seen_frame"] > self.reid_max_gap_frames:
                continue
            iou = self._bbox_iou(bbox, tr["bbox"])
            dist = self._center_distance(bbox, tr["bbox"])
            if iou >= self.reid_min_iou or dist <= self.reid_max_center_distance:
                if iou > best_iou or (abs(iou - best_iou) < 1e-9 and dist < best_dist):
                    best_sid, best_iou, best_dist = sid, iou, dist
        return best_sid

    def _resolve_stable_id(self, raw_pid, bbox, frame_idx, used_stable_ids):
        sid = None
        if raw_pid != -1 and raw_pid in self.raw_to_stable:
            sid = self.raw_to_stable[raw_pid]
            if sid in used_stable_ids:
                sid = None
        if sid is None:
            sid = self._match_recent_stable(bbox, frame_idx, used_stable_ids)
        if sid is None:
            sid = self._new_stable_id()
        if raw_pid != -1:
            self.raw_to_stable[raw_pid] = sid
        self.stable_tracks[sid] = {"bbox": bbox, "last_seen_frame": frame_idx}
        used_stable_ids.add(sid)
        return sid

    @staticmethod
    def _missing_signature(missing_set):
        return "|".join(sorted(missing_set)) if missing_set else "none"

    def _emit_with_cooldown(self, alerts, stable_id, status, ppe_status, bbox, frame_idx, missing_set):
        sig = (str(stable_id), status, self._missing_signature(missing_set))
        now = time.monotonic()
        last = self._last_emitted_alert_time.get(sig)
        if last is not None and (now - last) < self.alert_cooldown_seconds:
            return
        self._last_emitted_alert_time[sig] = now
        alerts.append({
            "person_id": stable_id,
            "status": status,
            "ppe_status": ppe_status,
            "bbox": bbox,
            "frame_num": frame_idx,
        })

    def _cleanup_stale(self, frame_idx):
        stale_sids = [
            sid for sid, tr in self.stable_tracks.items()
            if frame_idx - tr["last_seen_frame"] > self.state_ttl_frames
        ]
        for sid in stale_sids:
            self.stable_tracks.pop(sid, None)
            self.person_states.pop(sid, None)
            self.score_buffers.pop(sid, None)

        stale_raw = [rid for rid, sid in self.raw_to_stable.items() if sid not in self.stable_tracks]
        for rid in stale_raw:
            self.raw_to_stable.pop(rid, None)

    @staticmethod
    def _region_contains(region_bbox, item_bbox, pad=6):
        if region_bbox is None:
            return False
        rx1, ry1, rx2, ry2 = region_bbox
        cx, cy = PPELogicV9._bbox_center(item_bbox)
        return (rx1 - pad) <= cx <= (rx2 + pad) and (ry1 - pad) <= cy <= (ry2 + pad)

    @staticmethod
    def _regions_for_item(item_name: str):
        if item_name in {"helmet", "mask", "glasses"}:
            return ["head"]
        if item_name == "safety_vest":
            return ["torso"]
        if item_name == "boots":
            return ["feet"]
        if item_name == "glove":
            return ["hands"]
        return []

    @staticmethod
    def _draw_region(frame, region_bbox, label, color):
        if region_bbox is None:
            return
        x1, y1, x2, y2 = region_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _process_persons(self, frame, ppe_items, persons, frame_idx):
        detections_json = []
        alerts = []
        used_stable_ids = set()

        for person in persons:
            raw_pid = person.get("raw_pid", -1)
            bbox_tuple = tuple(person["bbox"])
            regions = person.get("regions", {})
            person_conf = float(person.get("conf", 0.0))
            px1, py1, px2, py2 = bbox_tuple

            stable_id = self._resolve_stable_id(raw_pid, bbox_tuple, frame_idx, used_stable_ids)
            state = self.person_states[stable_id]
            state["last_seen_frame"] = frame_idx

            for req_id in self.required_ppe_ids:
                item_name = self.ppe_class_map[req_id]
                region_keys = self._regions_for_item(item_name)
                region_bboxes = [regions.get(key) for key in region_keys if regions.get(key) is not None]
                if region_keys and not region_bboxes:
                    self.score_buffers[stable_id][req_id].append(None)
                    continue

                best_score = 0.0
                for cls_id, (x1, y1, x2, y2), _ in ppe_items:
                    if cls_id != req_id:
                        continue
                    item_bbox = (x1, y1, x2, y2)
                    if any(self._region_contains(region_bbox, item_bbox) for region_bbox in region_bboxes):
                        best_score = 1.0
                        break
                self.score_buffers[stable_id][req_id].append(best_score)

            avg_scores = {}
            gated_raw = {}
            gated_effective = {}

            for req_id in self.required_ppe_ids:
                item_name = self.ppe_class_map[req_id]
                valid_scores = [score for score in self.score_buffers[stable_id][req_id] if score is not None]
                avg_scores[item_name] = (sum(valid_scores) / len(valid_scores)) if valid_scores else None

                region_keys = self._regions_for_item(item_name)
                region_bboxes = [regions.get(key) for key in region_keys if regions.get(key) is not None]
                if region_keys and not region_bboxes:
                    gated_raw[item_name] = "na"
                    gated_effective[item_name] = "na"
                    continue

                score = avg_scores[item_name] or 0.0
                present = score >= 0.3
                if present:
                    gated_raw[item_name] = "yes"
                    gated_effective[item_name] = "yes"
                    state["last_yes_frame"][item_name] = frame_idx
                else:
                    gated_raw[item_name] = "no"
                    if frame_idx - state["last_yes_frame"].get(item_name, -(10 ** 9)) <= self.equipment_hold_frames:
                        gated_effective[item_name] = "yes"
                    else:
                        gated_effective[item_name] = "no"

            if self.show_regions:
                for region_name, region_bbox in regions.items():
                    self._draw_region(
                        frame,
                        region_bbox,
                        region_name.upper(),
                        _REGION_COLORS.get(region_name, (180, 180, 180)),
                    )

            effective_items = set(gated_effective.keys())
            essential_items = _ESSENTIAL_PPE.intersection(effective_items)
            non_essential_items = _NON_ESSENTIAL_PPE.intersection(effective_items)

            essential_missing = any(gated_effective.get(item) == "no" for item in essential_items)
            non_essential_missing = any(gated_effective.get(item) == "no" for item in non_essential_items)
            any_na = any(status == "na" for status in gated_effective.values())

            # Color policy:
            # RED    -> any essential PPE missing
            # YELLOW -> essentials are present, but some non-essential missing (or unresolved "na")
            # GREEN  -> all checked PPE items are present
            if essential_missing:
                box_color = (0, 0, 255)
            elif non_essential_missing or any_na:
                box_color = (0, 255, 255)
            else:
                box_color = (0, 255, 0)

            label = f"SID:{stable_id} {person_conf:.2f}"
            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(frame, label, (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

            if essential_missing or non_essential_missing:
                missing_items = [k for k, v in gated_effective.items() if v == "no"]
                if missing_items:
                    cv2.putText(
                        frame,
                        "Missing: " + ", ".join(missing_items),
                        (px1, py1 - 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
            elif any_na:
                missing_regions = sorted({
                    region_name
                    for req_id in self.required_ppe_ids
                    for region_name in self._regions_for_item(self.ppe_class_map[req_id])
                    if regions.get(region_name) is None
                })
                if missing_regions:
                    cv2.putText(
                        frame,
                        "Region missing: " + ", ".join(missing_regions),
                        (px1, py1 - 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

            detections_json.append({
                "person_id": stable_id,
                "avg_scores": avg_scores,
                "ppe_status": gated_effective,
                "raw_ppe_status": gated_raw,
                "bbox": [px1, py1, px2, py2],
                "regions": regions,
            })

            missing = {k for k, v in gated_effective.items() if v == "no"}
            is_safe = len(missing) == 0

            if state["alert_active"] and self.emit_violation_updates:
                previous_missing = set(state["last_missing"])
                if missing < previous_missing:
                    self._emit_with_cooldown(alerts, stable_id, "violation_updated", gated_effective, [px1, py1, px2, py2], frame_idx, missing)
                    state["last_missing"] = set(missing)

            if is_safe:
                state["bad_streak"] = 0
                state["good_streak"] += 1
                if state["pending"] and not state["alert_active"] and state["good_streak"] >= self.resolve_frames:
                    state["pending"] = False
                    state["pending_since"] = None
                if state["alert_active"] and state["good_streak"] >= self.resolve_frames:
                    self._emit_with_cooldown(alerts, stable_id, "violation_resolved", gated_effective, [px1, py1, px2, py2], frame_idx, set())
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
                        self._emit_with_cooldown(alerts, stable_id, "violation_started", gated_effective, [px1, py1, px2, py2], frame_idx, missing)
                        state["alert_active"] = True
                        state["pending"] = False
                        state["pending_since"] = None
                        state["last_missing"] = set(missing)

        return detections_json, alerts

    def process_frame_pose_items(self, frame, ppe_items, class_names, persons, frame_num=1):
        frame_idx = self._frame_index(frame_num)
        frame = frame.copy()
        detections_json, alerts = self._process_persons(frame, ppe_items, persons, frame_idx)

        for cls_id, (x1, y1, x2, y2), conf in ppe_items:
            name = class_names.get(cls_id, str(cls_id))
            color = self.class_colors.get(cls_id, (200, 200, 200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        self._cleanup_stale(frame_idx)
        return frame, detections_json, alerts if alerts else None
