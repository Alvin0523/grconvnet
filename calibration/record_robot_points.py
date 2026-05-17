from airbot_py.arm import AIRBOTPlay, RobotMode
import json
import time

# ========================
# Configuration
# ========================
AIRBOT_IP = "192.168.209.101"
AIRBOT_PORT = 50051
SAVE_FILE = "robot_poses.json"
MAX_POSES = 6

def main():
    poses = []
    
    try:
        # Initialize connection with timeout
        with AIRBOTPlay(url=AIRBOT_IP, port=AIRBOT_PORT) as robot:
            print(f"Successfully connected to {AIRBOT_IP}...")
            robot.switch_mode(RobotMode.GRAVITY_COMP)

        # Pose recording session
        print(f"\nPress Enter {MAX_POSES}x to capture poses (q to quit)")
        while len(poses) < MAX_POSES:
            input(f"Pose {len(poses)+1}/{MAX_POSES} - Press Enter to capture → ")

            # Capture and store raw terminal output
            with AIRBOTPlay(url=AIRBOT_IP, port=AIRBOT_PORT) as robot:
                pose = robot.get_end_pose()
                if not pose:
                    print("! Invalid pose - try again")
                    continue

            poses.append(pose)
            print(pose)

            # Update JSON file after each capture
            with open(SAVE_FILE, 'w') as f:
                json.dump(poses, f, indent=4)

        print(f"\n⚠️ Collected {MAX_POSES} poses - data saved to {SAVE_FILE}")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Current progress saved to JSON")

if __name__ == "__main__":
    main()
