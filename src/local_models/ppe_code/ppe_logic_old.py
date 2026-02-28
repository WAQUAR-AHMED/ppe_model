# import cv2
# from collections import defaultdict, deque

# class PPELogic:
#     def __init__(self, model_path=None):
#         # Thresholds per class
#         self.class_thresholds = {
#             0: 0.4,  # boots
#             1: 0.5,  # helmet
#             2: 0.3,  # no boots
#             3: 0.3,  # no helmet
#             4: 0.2,  # no vest
#             5: 0.5,  # person
#             6: 0.5   # vest
#         }

#         # Colors for drawing boxes
#         self.class_colors = {
#             0: (255, 0, 0),    # boots - blue
#             1: (0, 255, 255),  # helmet - yellow
#             2: (0, 0, 255),    # no boots - red
#             3: (255, 0, 255),  # no helmet - magenta
#             4: (0, 255, 0),    # no vest - green
#             5: (0, 165, 255),  # person - orange
#             6: (128, 0, 128)   # vest - purple
#         }

#         # rolling average buffers per person ID for smoother logic
#         self.score_buffers = defaultdict(lambda: defaultdict(lambda: deque(maxlen=30)))

#     def process_frame(self, result, frame_num=1):
#         frame = result.orig_img.copy()
#         detections_json = []

#         persons = []
#         others = []

#         # --- Step 1: detect persons ---
#         for box in result.boxes:
#             cls_id = int(box.cls.item())
#             conf = float(box.conf.item())
#             if conf < self.class_thresholds.get(cls_id, 0.5):
#                 continue

#             if cls_id == 5:  # person
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 pid = int(box.id.item()) if box.id is not None else -1
#                 persons.append((pid, (x1, y1, x2, y2)))

#                 # Draw person box
#                 label = f"ID:{pid} person {conf:.2f}"
#                 color = self.class_colors.get(cls_id, (255, 255, 255))
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, label, (x1, y1 - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         # --- Step 2: detect PPE items ---
#         if persons:
#             for box in result.boxes:
#                 cls_id = int(box.cls.item())
#                 conf = float(box.conf.item())
#                 if conf < self.class_thresholds.get(cls_id, 0.5) or cls_id == 5:
#                     continue

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 others.append((cls_id, (x1, y1, x2, y2)))

#                 # Draw PPE box
#                 label = f"{result.names[cls_id]} {conf:.2f}"
#                 color = self.class_colors.get(cls_id, (255, 255, 255))
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, label, (x1, y1 - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         # --- Step 3: PPE logic per person ---
#         for pid, (px1, py1, px2, py2) in persons:
#             scores = defaultdict(float)

#             for cls_id, (x1, y1, x2, y2) in others:
#                 inside = (x1 > px1 and y1 > py1 and x2 < px2 and y2 < py2)
#                 conf_score = 1.0 if inside else 0.0
#                 scores[cls_id] = conf_score
#                 self.score_buffers[pid][cls_id].append(conf_score)

#             # Compute averages
#             avg_scores = {}
#             for cid, buf in self.score_buffers[pid].items():
#                 avg_scores[result.names[cid]] = sum(buf) / len(buf)

#             avg_scores["person"] = 1.0

#             # Determine yes/no for PPE
#             comparisons = {
#                 "boots": "yes" if avg_scores.get("boots", 0) > avg_scores.get("no boots", 0) else "no",
#                 "helmet": "yes" if avg_scores.get("helmet", 0) > avg_scores.get("no helmet", 0) else "no",
#                 "vest": "yes" if avg_scores.get("vest", 0) > avg_scores.get("no vest", 0) else "no"
#             }

#             detections_json.append({
#                 "person_id": pid,
#                 "avg_scores": avg_scores,
#                 "ppe_status": comparisons,
#                 "bbox": [px1, py1, px2, py2]
#             })

#             # Draw PPE summary below person
#             summary = f"H:{comparisons['helmet']} V:{comparisons['vest']} B:{comparisons['boots']}"
#             cv2.putText(frame, summary, (px1, py2 + 20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#         return frame, detections_json



import cv2
from collections import defaultdict, deque

class PPELogic:
    def __init__(self, model_path=None):

        # Thresholds per class
        self.class_thresholds = {
            0: 0.4,  # boots
            1: 0.5,  # helmet
            2: 0.3,  # no boots
            3: 0.3,  # no helmet
            4: 0.2,  # no vest
            5: 0.5,  # person
            6: 0.5   # vest
        }

        # Colors for drawing
        self.class_colors = {
            0: (255, 0, 0),
            1: (0, 255, 255),
            2: (0, 0, 255),
            3: (255, 0, 255),
            4: (0, 255, 0),
            5: (0, 165, 255),
            6: (128, 0, 128)
        }

        # Rolling average buffer
        self.score_buffers = defaultdict(lambda: defaultdict(lambda: deque(maxlen=30)))

        # Alert flag per person
        # True  → alert already sent (safe)
        # False → no alert sent or currently violating
        self.alert_sent = defaultdict(lambda: True)


    def process_frame(self, result, frame_num=1):
        frame = result.orig_img.copy()
        detections_json = []
        alerts = []

        persons = []
        others = []

        # ------------------------ PERSON DETECTION ------------------------
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            if conf < self.class_thresholds.get(cls_id, 0.5):
                continue

            if cls_id == 5:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pid = int(box.id.item()) if box.id is not None else -1
                persons.append((pid, (x1, y1, x2, y2)))

                label = f"ID:{pid} person {conf:.2f}"
                color = self.class_colors.get(cls_id, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ------------------------ PPE DETECTION ------------------------
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
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ------------------------ PPE LOGIC PER PERSON ------------------------
        for pid, (px1, py1, px2, py2) in persons:

            scores = defaultdict(float)

            # Assign PPE inside person box
            for cls_id, (x1, y1, x2, y2) in others:
                inside = (x1 > px1 and y1 > py1 and x2 < px2 and y2 < py2)
                score = 1.0 if inside else 0.0
                scores[cls_id] = score
                self.score_buffers[pid][cls_id].append(score)

            # Rolling averages
            avg_scores = {}
            for cid, buf in self.score_buffers[pid].items():
                avg_scores[result.names[cid]] = sum(buf) / len(buf)

            avg_scores["person"] = 1.0

            # PPE status
            comparisons = {
                "boots":  "yes" if avg_scores.get("boots", 0)  > avg_scores.get("no boots", 0)  else "no",
                "helmet": "yes" if avg_scores.get("helmet", 0) > avg_scores.get("no helmet", 0) else "no",
                "vest":   "yes" if avg_scores.get("vest", 0)   > avg_scores.get("no vest", 0)   else "no"
            }

            detections_json.append({
                "person_id": pid,
                "avg_scores": avg_scores,
                "ppe_status": comparisons,
                "bbox": [px1, py1, px2, py2]
            })

            summary = f"H:{comparisons['helmet']} V:{comparisons['vest']} B:{comparisons['boots']}"
            cv2.putText(frame, summary, (px1, py2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # ------------------------- ALERT LOGIC -------------------------
            helmet_ok = comparisons["helmet"] == "yes"
            vest_ok   = comparisons["vest"]   == "yes"
            boots_ok  = comparisons["boots"]  == "yes"

            is_safe = helmet_ok and vest_ok and boots_ok
            previous_alert_state = self.alert_sent[pid]

            if not is_safe:
                # ❗ Violation started AND alert not sent → SEND ALERT ONCE
                if previous_alert_state:  
                    alerts.append({
                        "person_id": pid,
                        "status": "violation",
                        "ppe_status": comparisons,
                        "bbox": [px1, py1, px2, py2]
                    })

                # Mark person as "in violation"
                self.alert_sent[pid] = False

            else:
                # Person fully safe → reset alert
                self.alert_sent[pid] = True

        return frame, detections_json, alerts if alerts else None

