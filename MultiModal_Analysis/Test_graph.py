import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Process, Queue
import time

# 비디오 파일 불러오는 경로 
videos = [
    'C:/Users/user/Downloads/Face Video/Face_1W_A1_S2.mp4',
    'C:/Users/user/Downloads/Face Video/Face_1W_A2_S2.mp4',
    'C:/Users/user/Downloads/Face Video/A3_result/Face_1W_A3_S2.mp4',
    'C:/Users/user/Downloads/Face Video/A4_result/Face_1W_A4_S2.mp4'
]

# CSV 파일 불러오는 경로 
csv_files = [
    'C:/Users/user/Downloads/Face Video/A1_result/head_pose_coordinates_delta_A1.csv',
    'C:/Users/user/Downloads/Face Video/A2_result/head_pose_coordinates_delta_A2.csv',
    'C:/Users/user/Downloads/Face Video/A3_result/head_pose_coordinates_delta_A3.csv',
    'C:/Users/user/Downloads/Face Video/A4_result/head_pose_coordinates_delta_A4.csv'
]

# 특정 보여줄 시간 
start_seconds = 240
end_seconds = 270

# 영상을 고정된 너비와 높이로 보여주기 위한 것 
fixed_width = 400
fixed_height = 240

# 전체 초
total_seconds = end_seconds - start_seconds

# 그래프를 고정된 너비와 높이로 보여주기 위한 것
graph_width = 10
graph_height = 5

# 비디오를 일어와서, 보여주기 위한 부분. 
def display_video(video_path, window_index, fps_queue, stop_queue):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    fps_queue.put(fps)
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)
    total_frames = end_frame - start_frame
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    window_name = f'Video {window_index + 1}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, fixed_width, fixed_height)
    cv2.moveWindow(window_name, 100, window_index * (fixed_height + 5))

    frame_num = 0
    
    while cap.isOpened() and frame_num < total_frames:
        start_time = time.time()  # 시작 시간 기록
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (fixed_width, fixed_height))
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1
        
        elapsed_time = time.time() - start_time
        time_to_wait = max(1/fps - elapsed_time, 0)
        
        time.sleep(time_to_wait)  # 다음 프레임까지 대기
    
    cap.release()
    cv2.destroyWindow(window_name)
    stop_queue.put(True)

def display_graphs(csv_files, fps_queue, stop_queue):
    fps = fps_queue.get()
    total_frames = int((end_seconds - start_seconds) * fps)
    fig, axes = plt.subplots(len(csv_files), 1, figsize=(graph_width, graph_height * len(csv_files)))

    red_lines = []
    for i, csv_file in enumerate(csv_files):
        data = pd.read_csv(csv_file)
        start_frame = int(start_seconds * fps)
        end_frame = int(end_seconds * fps)
        data_trimmed = data['X'][start_frame:end_frame]

        ax = axes[i]
        years = np.arange(0, total_seconds, 1 / fps)
        ax.plot(years, data_trimmed, lw=2)

        y_min = np.floor(data_trimmed.min())
        y_max = np.ceil(data_trimmed.max())
        ax.set_ylim(y_min, y_max)

        xticks = np.arange(0, total_seconds + 1, 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels((xticks + start_seconds).astype(int))
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlim(0, total_seconds)

        red_line = ax.axvline(x=0, color='r')
        red_lines.append(red_line)

        graph_title = f"A{i + 1} graph"
        ax.set_title(graph_title, pad=20)

    plt.tight_layout()
    plt.ion()
    plt.show()

    while plt.fignum_exists(fig.number):
        if not stop_queue.empty():
            break
        for frame_num in range(total_frames):
            if not stop_queue.empty():
                break
            current_time_sec = frame_num / fps
            for red_line in red_lines:
                red_line.set_xdata([current_time_sec])
            plt.draw()
            plt.pause(1 / fps)
    
    plt.close(fig)

if __name__ == "__main__":
    fps_queue = Queue()
    stop_queue = Queue()
    processes = []
    for i in range(len(videos)):
        p_video = Process(target=display_video, args=(videos[i], i, fps_queue, stop_queue))
        processes.append(p_video)
        p_video.start()

    p_graphs = Process(target=display_graphs, args=(csv_files, fps_queue, stop_queue))
    processes.append(p_graphs)
    p_graphs.start()

    for p in processes:
        p.join()

    cv2.destroyAllWindows()


# 영상이랑 그래프 전체 다 보이고, 깔끔하게 나온 것. = 후보 1

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from multiprocessing import Process, Queue

# # 비디오 파일 경로
# videos = [
#     'C:/Users/user/Downloads/Face Video/Face_1W_A1_S2.mp4',
#     'C:/Users/user/Downloads/Face Video/Face_1W_A2_S2.mp4',
#     'C:/Users/user/Downloads/Face Video/A3_result/Face_1W_A3_S2.mp4',
#     'C:/Users/user/Downloads/Face Video/A3_result/Face_1W_A4_S2.mp4'
# ]

# # CSV 파일 경로
# csv_files = [
#     'C:/Users/user/Downloads/Face Video/A1_result/head_pose_coordinates_delta_A1.csv',
#     'C:/Users/user/Downloads/Face Video/A2_result/head_pose_coordinates_delta_A2.csv',
#     'C:/Users/user/Downloads/Face Video/A3_result/head_pose_coordinates_delta_A3.csv',
#     'C:/Users/user/Downloads/Face Video/A4_result/head_pose_coordinates_delta_A4.csv'
# ]

# start_seconds = 240
# end_seconds = 270
# fixed_width = 400
# fixed_height = 240
# total_seconds = end_seconds - start_seconds
# graph_width = 10
# graph_height = 5

# def display_video(video_path, window_index, fps_queue):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     fps_queue.put(fps)
#     start_frame = int(start_seconds * fps)
#     end_frame = int(end_seconds * fps)
#     total_frames = end_frame - start_frame
    
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
#     window_name = f'Video {window_index + 1}'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(window_name, fixed_width, fixed_height)
#     cv2.moveWindow(window_name, 100, window_index * (fixed_height + 50))

#     frame_num = 0
#     while cap.isOpened() and frame_num < total_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 비디오 프레임을 고정된 크기로 리사이즈
#         frame = cv2.resize(frame, (fixed_width, fixed_height))
        
#         # 비디오 프레임을 OpenCV 창에 표시
#         cv2.imshow(window_name, frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         frame_num += 1

#     cap.release()
#     cv2.destroyWindow(window_name)

# def display_graphs(csv_files, fps_queue):
#     fps = fps_queue.get()  # Get the FPS value from the queue
#     total_frames = int((end_seconds - start_seconds) * fps)
#     fig, axes = plt.subplots(len(csv_files), 1, figsize=(graph_width, graph_height * len(csv_files)))

#     red_lines = []
#     for i, csv_file in enumerate(csv_files):
#         data = pd.read_csv(csv_file)
#         start_frame = int(start_seconds * fps)
#         end_frame = int(end_seconds * fps)
#         data_trimmed = data['X'][start_frame:end_frame]

#         ax = axes[i]
#         years = np.arange(0, total_seconds, 1/fps)
#         ax.plot(years, data_trimmed, lw=2)

#         y_min = np.floor(data_trimmed.min())
#         y_max = np.ceil(data_trimmed.max())
#         ax.set_ylim(y_min, y_max)

#         xticks = np.arange(0, total_seconds + 1, 1)
#         ax.set_xticks(xticks)
#         ax.set_xticklabels((xticks + start_seconds).astype(int))
#         ax.tick_params(axis='x', rotation=45)
#         ax.set_xlim(0, total_seconds)

#         red_line = ax.axvline(x=0, color='r')
#         red_lines.append(red_line)

#         # 그래프 제목 설정
#         graph_title = f"A{i + 1} graph"
#         ax.set_title(graph_title)

#     plt.tight_layout()
#     plt.ion()
#     plt.show()

#     while True:
#         for frame_num in range(total_frames):
#             current_time_sec = frame_num / fps
#             for red_line in red_lines:
#                 red_line.set_xdata([current_time_sec])
#             plt.draw()
#             plt.pause(1/fps)

# if __name__ == "__main__":
#     fps_queue = Queue()
#     processes = []
#     for i in range(len(videos)):
#         p_video = Process(target=display_video, args=(videos[i], i, fps_queue))
#         processes.append(p_video)
#         p_video.start()

#     p_graphs = Process(target=display_graphs, args=(csv_files, fps_queue))
#     processes.append(p_graphs)
#     p_graphs.start()

#     for p in processes:
#         p.join()

#     cv2.destroyAllWindows()






# 영상이랑 그래프가 동시에 보일 수 있도록 한 코드 


# import cv2
# import pandas as pd
# import matplotlib.pyplot as plt
# from multiprocessing import Process

# # 비디오 파일 경로
# videos = [
#     'C:/Users/user/Downloads/Face Video/Face_1W_A1_S2.mp4',
#     'C:/Users/user/Downloads/Face Video/Face_1W_A2_S2.mp4',
#     'C:/Users/user/Downloads/Face Video/A3_result/Face_1W_A3_S2.mp4',
#     'C:/Users/user/Downloads/Face Video/A4_result/Face_1W_A4_S2.mp4'
# ]

# # CSV 파일 경로
# csv_files = [
#     'C:/Users/user/Downloads/Face Video/A1_result/head_pose_coordinates_delta_A1.csv',
#     'C:/Users/user/Downloads/Face Video/A2_result/head_pose_coordinates_delta_A2.csv',
#     'C:/Users/user/Downloads/Face Video/A3_result/head_pose_coordinates_delta_A3.csv',
#     'C:/Users/user/Downloads/Face Video/A4_result/head_pose_coordinates_delta_A4.csv'
# ]

# start_seconds = 240
# end_seconds = 270
# fixed_width = 400
# fixed_height = 240

# def display_video(video_path, window_index):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     start_frame = int(start_seconds * fps)
#     end_frame = int(end_seconds * fps)
#     total_frames = end_frame - start_frame
    
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
#     window_name = f'Video {window_index + 1}'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(window_name, fixed_width, fixed_height)
#     cv2.moveWindow(window_name, 100, window_index * (fixed_height + 10))

#     frame_num = 0
#     while cap.isOpened() and frame_num < total_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 비디오 프레임을 고정된 크기로 리사이즈
#         frame = cv2.resize(frame, (fixed_width, fixed_height))
        
#         # 비디오 프레임을 OpenCV 창에 표시
#         cv2.imshow(window_name, frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         frame_num += 1

#     cap.release()
#     cv2.destroyWindow(window_name)

# def display_graph(csv_file, window_index):
#     data = pd.read_csv(csv_file)
#     start_frame = int(start_seconds * 30)  # assuming 30 fps
#     end_frame = int(end_seconds * 30)
#     data_trimmed = data['X'][start_frame:end_frame]

#     fig, ax = plt.subplots(figsize=(5, 2.4))
#     ax.plot(data_trimmed)
#     ax.set_title(f'Graph {window_index + 1}')
#     ax.set_xlabel('Frames')
#     ax.set_ylabel('Value')
#     ax.set_ylim(data_trimmed.min() - 1, data_trimmed.max() + 1)
#     ax.grid(True)
    
#     # 그래프 창의 위치 설정
#     manager = plt.get_current_fig_manager()
#     manager.window.wm_geometry(f"+600+{window_index * (fixed_height + 10)}")
#     plt.show()

# if __name__ == "__main__":
#     # 프로세스 생성 및 시작
#     processes = []
#     for i in range(len(videos)):
#         p_video = Process(target=display_video, args=(videos[i], i))
#         p_graph = Process(target=display_graph, args=(csv_files[i], i))
#         processes.append(p_video)
#         processes.append(p_graph)
#         p_video.start()
#         p_graph.start()

#     # 프로세스가 종료될 때까지 대기
#     for p in processes:
#         p.join()

#     cv2.destroyAllWindows()


# 영상만 쭉 세로로 보일 수 있도록 한 코드 

# import cv2
# from multiprocessing import Process

# # 비디오 파일 경로
# videos = [
#     'C:/Users/user/Downloads/Face Video/Face_1W_A1_S2.mp4',
#     'C:/Users/user/Downloads/Face Video/Face_1W_A2_S2.mp4',
#     'C:/Users/user/Downloads/Face Video/A3_result/Face_1W_A3_S2.mp4',
#     'C:/Users/user/Downloads/Face Video/A4_result/Face_1W_A4_S2.mp4'
# ]

# start_seconds = 240
# end_seconds = 270
# fixed_width = 400
# fixed_height = 240

# def display_video(video_path, window_index):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     start_frame = int(start_seconds * fps)
#     end_frame = int(end_seconds * fps)
#     total_frames = end_frame - start_frame
    
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
#     window_name = f'Video {window_index + 1}'
#     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#     cv2.resizeWindow(window_name, fixed_width, fixed_height)
#     cv2.moveWindow(window_name, 100, window_index * (fixed_height + 10))

#     frame_num = 0
#     while cap.isOpened() and frame_num < total_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # 비디오 프레임을 고정된 크기로 리사이즈
#         frame = cv2.resize(frame, (fixed_width, fixed_height))
        
#         # 비디오 프레임을 OpenCV 창에 표시
#         cv2.imshow(window_name, frame)
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         frame_num += 1

#     cap.release()
#     cv2.destroyWindow(window_name)

# if __name__ == "__main__":
#     # 프로세스 생성 및 시작
#     processes = []
#     for i in range(len(videos)):
#         p = Process(target=display_video, args=(videos[i], i))
#         processes.append(p)
#         p.start()

#     # 프로세스가 종료될 때까지 대기
#     for p in processes:
#         p.join()

#     cv2.destroyAllWindows()











