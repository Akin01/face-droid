import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

current_time = 0
previous_time = 0

while True:
    res, frame = cap.read()

    height_frame, width_frame, channel = frame.shape

    frame = cv2.flip(frame, 1)
    results = hands.process(frame)

    if res:
        if results.multi_hand_landmarks:
            for hand_ldk in results.multi_hand_landmarks:
                for idx, dim in enumerate(hand_ldk.landmark):
                    h, w, c = frame.shape
                    xHand, yHand = int(dim.x * w), int(dim.y * h)

                    if (idx % 4 == 0) and (idx != 0):

                        cv2.circle(frame, (xHand, yHand), 10, (0, 0, 0), 4)

                        if (390 < xHand < 420) and (350 < yHand < 380):
                            cv2.rectangle(frame,
                                          (int((width_frame / 2)) - 220, int((height_frame / 2)) - 50),
                                          (int((width_frame / 2)) - 100, int((height_frame / 2)) + 20),
                                          (85, 41, 39), -1)
                            cv2.rectangle(frame,
                                          (int((width_frame / 2)) - 220, int((height_frame / 2)) - 50),
                                          (int((width_frame / 2)) - 100, int((height_frame / 2)) + 20),
                                          (31, 16, 14), 2)
                            cv2.putText(frame, "KIRI", (int((width_frame / 2)) - 200, int((height_frame / 2))),
                                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

                        elif (210 < xHand < 260) and (350 < yHand < 380):
                            cv2.rectangle(frame,
                                          (int((width_frame / 2)) + 80, int((height_frame / 2)) - 50),
                                          (int((width_frame / 2)) + 270, int((height_frame / 2)) + 20),
                                          (85, 41, 39), -1)
                            cv2.rectangle(frame,
                                          (int((width_frame / 2)) + 80, int((height_frame / 2)) - 50),
                                          (int((width_frame / 2)) + 270, int((height_frame / 2)) + 20),
                                          (31, 16, 14), 2)
                            cv2.putText(frame, "KANAN", (int((width_frame / 2)) + 100, int((height_frame / 2))),
                                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

                mp_drawing.draw_landmarks(frame, hand_ldk, mp_hands.HAND_CONNECTIONS)

    current_time = time.time()
    fps = int(1 / (current_time - previous_time))

    previous_time = current_time

    cv2.putText(frame, f"{fps} fps", (int((width_frame / 2)) - 200, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.putText(frame, "Ainul Yaqin", (int((width_frame / 2)) + 50, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Hand Landmarks Detections", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
