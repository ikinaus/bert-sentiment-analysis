# How to run the project

Run through the "train_notebook.ipynb" and create all necessary files (.pth, *.onnx, probably*.onnx.data)

then:

```bash
docker-compose up -d --build
```

*train_notebook.ipynb* - model training pipline

Testing the running Triton instance: Run test_initon.ipynb for gRPC requests to the server.
