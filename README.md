# Real_Time_Object_Detection_Under_Diverse_Visibility_Conditions
real-time object detection framework designed to handle diverse visibility conditions efficiently


This system will operate as follow - continuous video frames and feeds them into a lightweight visibility classification model, which quickly identifies the current visibility condition (e.g., clear, rain, fog). The condition classification accrues every few seconds, as the assumptions that visibility condition doesn’t change in a smaller time frame. Based on this classification, the system dynamically selects the corresponding YOLOv8 sub condition weights, optimized for that specific condition. The YOLO model then detects objects (cars, pedestrians, etc.) in the frame, generating bounding boxes and confidence scores. 

<img width="271" alt="image" src="https://github.com/user-attachments/assets/3ddc11f1-d471-4659-9f43-0e11a99bfebf" />

Complete system pipeline output. In the left photo, the visibility condition classification, in the right photo, yolov8 output with fog weights:

<img width="278" alt="image" src="https://github.com/user-attachments/assets/7d900c50-219e-4994-a5bb-18eab49bce26" />

Compared performance of the YOLOv8 model when loaded with weights trained on each specific visibility condition versus weights trained on the combined dataset - 

<img width="285" alt="image" src="https://github.com/user-attachments/assets/1ff6d247-c71c-4064-865c-7677e0636b57" />

On top (green frame) results of the YOLOv8 model using fog-specific weights. Bottom (red frame) results using weights trained on all conditions.


