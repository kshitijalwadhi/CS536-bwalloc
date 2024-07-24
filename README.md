# CS536-bwalloc

## Abstract
This study addresses the intricate challenge of bandwidth allocation for multi-stream video analytics within the context of edge computing. The focal point of our investigation involves a scenario where N cameras transmit streams to a single-edge server executing real-time object detection tasks. Two distinct paradigms are explored: one involves the server dropping frames from clients based on bandwidth considerations, while the other entails the server engaging in communication with the client to dynamically adjust its utilization. Furthermore, we investigate the nuanced trade-off between video resolution and bandwidth constraints, carefully considering the impact on the performance of our object detection algorithm. Our approach employs a diverse set of techniques, with a specific emphasis on leveraging mixed-integer linear programming to dynamically fine-tune bandwidth allocation. This ensures the optimal balance between video analytics accuracy and efficient management of network resources. The research presented herein contributes to the field by introducing a novel algorithm that adeptly navigates the complexities of balancing video quality and bandwidth limitations.

## Usage
### First, create a virtual environment and install the necessary packages:

```
python3 -m venv .venv
source .venv_agbench/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Second, compile the grpc proto by running:

Follow this for installing grpc: `https://grpc.io/docs/languages/python/quickstart/`

`python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. object_detection.proto`

### Third, run the server as: 

`python3 server.py`

### Finally, run clients as:
`python3 client.py --fps <fps>`
