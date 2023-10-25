# CS526-bwalloc


## To compile the grpc proto:

Follow this for installing grpc: `https://grpc.io/docs/languages/python/quickstart/`

`python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. object_detection.proto`