import gradio as gr
import cv2
from ultralytics import YOLO, solutions
import tempfile
#from ultralytics import YOLOv10
#import numpy as np
#from shapely.geometry import Polygon
#from shapely.geometry.point import Point
from ultralytics.solutions.object_counter import ObjectCounter


def yolo_inference(image, video, model_id, image_size, conf_threshold):
    #model = YOLOv10.from_pretrained(f'jameslahm/{model_id}')
    #model = YOLO("yolov8n.pt")
    model = YOLO(f'{model_id}.pt')
    
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None
    else:
        video_path = tempfile.mktemp(suffix=".webm")
        with open(video_path, "wb") as f:
            with open(video, "rb") as g:
                f.write(g.read())

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        line_points = [(0, int(frame_height/2)), (frame_width, int(frame_height/2))]
        poly_points = [(0, frame_height // 3), (frame_width, frame_height // 3), (frame_width, frame_height * 2 // 3), (0, frame_height * 2 // 3)]
        #poly_points = [(0, frame_height // 10), (frame_width, frame_height // 10), (frame_width, frame_height // 3), (0, frame_height // 3)]
        
        split_line_y = frame_height // 2

        
        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))
        
        """
        output_video_path = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        """
        
        #out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        
        # Init Object Counter
        counter = ObjectCounter(
            view_img=True,
            reg_pts=poly_points,
            #reg_pts=line_points,
            names=model.names,
            view_in_counts=True,
            view_out_counts=True,
            draw_tracks=False,
            line_thickness=2,
        )       
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            #results = model.track(source=frame, persist = True, show = True, imgsz=image_size, conf=conf_threshold)
            tracks = model.track(source=frame, persist=True, show=False)
            #annotated_frame = results[0].plot()
            frame = counter.start_counting(frame, tracks)
            out.write(frame) 

        cap.release()
        out.release()

        return None, output_video_path

def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Video", visible=True)
                image = gr.Image(type="pil", label="Image", visible=False)
                input_type = gr.Radio(
                    choices=["Video"],
                    value="Video",
                    label="Input Type",
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolov8n",
                        "yolov8s",
                        "yolov8m",
                        "yolov8l",
                        "yolov8x",
                    ],
                    value="yolov8n",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolo_infer = gr.Button(value="Detect")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=False)
                output_video = gr.Video(label="Annotated Video", visible=True)

        def update_visibility(input_type):
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)

            return image, video, output_image, output_video

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        def run_inference(image, video, model_id, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return yolo_inference(image, None, model_id, image_size, conf_threshold)
            else:
                return yolo_inference(None, video, model_id, image_size, conf_threshold)


        yolo_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Real-Time End-to-End Object Detection
    </h1>
    """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch(server_name="0.0.0.0", server_port=8000)