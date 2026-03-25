import cv2
from ultralytics import YOLO

# load trained model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("traffic1.mp4")

# counting line position
line_y = 1500

# vehicle types
vehicle_counts = {
    "car":0,
    "truck":0,
    "bus":0,
    "motorcycle":0,
    "bicycle":0
}

counted_ids = set()

cv2.namedWindow("Traffic Analysis", cv2.WINDOW_NORMAL)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.5)

    # draw counting line
    cv2.line(frame,(0,line_y),(frame.shape[1],line_y),(0,0,255),3)

    for r in results:

        boxes = r.boxes

        for box in boxes:

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name in vehicle_counts:

                x1,y1,x2,y2 = map(int, box.xyxy[0])

                # tracking ID
                track_id = int(box.id[0]) if box.id is not None else None

                center_y = (y1+y2)//2

                # counting when crossing line
                if track_id not in counted_ids and center_y > line_y:

                    counted_ids.add(track_id)
                    vehicle_counts[class_name] += 1

                # draw box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(frame,f"{class_name} ID:{track_id}",
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,(0,255,0),2)

    # total vehicles
    total = sum(vehicle_counts.values())

    # traffic density
    frame_area = frame.shape[0] * frame.shape[1]
    density = total / frame_area

    # show stats
    cv2.putText(frame,f"Total Vehicles: {total}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.putText(frame,f"Density: {density:.6f}",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    # resize to avoid zoom
    display = cv2.resize(frame,(1000,600))

    cv2.imshow("Traffic Analysis",display)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()


# -------- vehicle composition --------
print("\nVehicle Composition\n")

total = sum(vehicle_counts.values())

for v,c in vehicle_counts.items():

    if total > 0:
        percent = (c/total)*100
    else:
        percent = 0

    print(f"{v} : {c} ({percent:.2f}%)")
