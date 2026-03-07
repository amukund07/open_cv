import cv2

import hand_module as hm

cap = cv2.VideoCapture(0)

# Custom settings
detector = hm.HandDetector(
    num_hands=1,
    connections=[(0,8), (4,12)]
)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not success:
        break

    frame, hands = detector.find_hands(
        frame,
        point_color=(0,0,255),
        line_color=(0,255,255),
        point_radius=10
    )

    if len(hands) > 0:
        hand = hands[0]
        distance = detector.find_distance(hand, 4, 8)
        print("Distance:", distance)

    cv2.imshow("New Project", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()