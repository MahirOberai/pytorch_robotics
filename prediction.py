import argparse
from os import close
from sim import Action, WallESim
import pybullet as p
import torch
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
import cv2

CLOSE_DISTANCE_THRESHOLD = 0.3

class ImgProcessingActionPredictor:
    def __init__(self):
        pass

    def predict_action(self, img):
        #action = Action.RIGHT
        # TODO: 
        # ===============================================
        # ===============================================
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, threshold_img = cv2.threshold(grayscale_img, 50, 255, cv2.THRESH_BINARY)

        for row in threshold_img:
            threshold_idx = np.where(row==255)
            if len(threshold_idx[0]):
                min_idx = np.amin(threshold_idx)
                max_idx = np.amax(threshold_idx)        

                idx_diff = max_idx - min_idx
                mid_idx = min_idx + np.floor((idx_diff)/2)
                if mid_idx > grayscale_img.shape[0]/2:
                    action = Action.RIGHT
                #else:
                    #action = Action.LEFT

                if grayscale_img.shape[0]/2-(idx_diff/3) <= mid_idx <= grayscale_img.shape[0]/2+(idx_diff/3):
                    action = Action.FORWARD
            
            else:
                action = Action.RIGHT
        return action


class ImitationLearningActionPredictor:
    def __init__(self, model_path, transform=None):
        # TODO: Load model.
        # ===============================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.transform = transform
        # ===============================================

    def predict_action(self, img):
        action = Action.FORWARD

        # TODO: 
        # ===============================================
        # ===============================================
        #
        if self.transform:
            img = self.transform(img)

        img = img.unsqueeze(0).to(self.device)
        self.model.eval()
        output = self.model(img)
        _, pred = torch.max(output, 1)
        action_idx = pred.item()
        action = Action(action_idx)
        
        return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser("HW4: Testing line following algorithms")
    parser.add_argument("--use_imitation_learning", "-uip", action="store_true",
                        help="Algorithm to use: 0->image processing, 1->trained model")
    parser.add_argument("--map_path", "-m", type=str, default="maps/test/map1",
                        help="path to map directory. eg: maps/test/map2")
    parser.add_argument("--model_path", type=str, default="extra_following_model.pth", #"following_model.pth"
                        help="Path to trained imitation learning based action predictor model")
    args = parser.parse_args()

    env = WallESim(args.map_path, load_landmarks=True)

    if args.use_imitation_learning:
        # TODO: Provide transform arguments if any to the constructor
        # =================================================================
        actionPredictor = ImitationLearningActionPredictor(args.model_path, transform=transforms.ToTensor())
        # =================================================================
    else:
        actionPredictor = ImgProcessingActionPredictor()

    landmarks_reached = np.zeros(len(env.landmarks), dtype=np.bool)
    assert len(landmarks_reached) != 0
    iteration = 1
    while True:
        #env.set_landmarks_visibility(False)
        rgbImg = env.get_robot_view()
        #env.set_landmarks_visibility(True)
        action = actionPredictor.predict_action(rgbImg)
        env.move_robot(action)

        position, _ = p.getBasePositionAndOrientation(env.robot_body_id)
        distance_from_landmarks = np.linalg.norm(env.landmarks - position, axis=1)
        closest_landmark_index = np.argmin(distance_from_landmarks)
        if distance_from_landmarks[closest_landmark_index] < CLOSE_DISTANCE_THRESHOLD and not landmarks_reached[closest_landmark_index]:
            landmarks_reached[closest_landmark_index] = True
        print(
            f"[{iteration}] {np.sum(landmarks_reached)} / {len(landmarks_reached)} landmarks reached. "
            f"{distance_from_landmarks[closest_landmark_index]:.2f} distance away from nearest landmark"
        )
        if np.all(landmarks_reached):
            print("All landmarks reached!")
            break
        iteration += 1
