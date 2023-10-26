import argparse


def send_video(client_fps, client_packet_drop_rate):
    pass

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fps",
        type=int,
        required=False,
        help="Frame Rate for Client",
        metavar="",
        default=30
    )
    parser.add_argument(
        "packet_drop_rate",
        type=str,
        required=False,
        help="Packet Drop Rate for Client",
        metavar="",
    )

if __name__ == "__main__":
    args = get_args()
    client_fps = args.fps
    client_packet_drop_rate = args.packet_drop_rate
    send_video(client_fps, client_packet_drop_rate)