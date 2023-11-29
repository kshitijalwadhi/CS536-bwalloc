# CS536-bwalloc

## Servers available:
1. FCFS Server
2. Priority Queue with Normalization Factor. Queue ordered in descending order.
3. Priority Queue with Normalization Factor. Queue ordered in ascending order.

## To compile the grpc proto:

Follow this for installing grpc: `https://grpc.io/docs/languages/python/quickstart/`

`python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. object_detection.proto`
