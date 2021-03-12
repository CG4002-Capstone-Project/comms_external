# comms_external

# external comms

## Instruction
- `Ultra96_Server.py` runs on Ultra96 and receives data from different dancers' laptop to pass to ML model for inference
- Port number is fixed for dancer 0,1,2 and are set as 9091,9092,9093 respectively
- Password is also fixed and is set as 1234123412341234 (default)
- `eval_client.py` runs on Ultra96 and sends data back to evaluation server The client sending data back to evaluation server

## Setup
```
pip install -r requirements.txt
```

## Run 
- Run in with model in debug mode
```
python3 Ultra96_Server.py --dancer_id 0 --debug True --model_type dnn --model_path ./dnn_model.pth --scaler_path ./dnn_std_scaler.bin
python3 Ultra96_Server.py --dancer_id 0 --debug True --model_type svc --model_path ./svc_model.pth --scaler_path ./svc_std_scaler.bin
```
- Run in production mode
```
python3 main.py --dancer_id 0 --production True 
```
- Set the verbose flag to print statements and secret_key to change password

## Linting
```
./fix_lint.sh
```

## Access Ultra96
1. You need to ssh into sunfire (for students):
```
ssh -l nusnet_id sunfire.comp.nus.edu.sg 
```
2. From Sunfire, you can access the boards:
```
ssh -l xilinx <IP address of your group's board>
makerslab-fpga-18    f8:f0:05:dd:e9:78    137.132.86.241
```
### Examples
- Access Ultra96 via SSH
```
ssh zenghao@sunfire.comp.nus.edu.sg
ssh -l xilinx 137.132.86.241
```
- Transfer files to Ultra96 via SSH
```
scp eval_server.py xilinx@137.132.86.241:~/
scp eval_server.py zenghao@sunfire.comp.nus.edu.sg:~/
scp Ultra96_Server.py xilinx@137.132.86.241:~/
scp Ultra96_Server.py zenghao@sunfire.comp.nus.edu.sg:~/
```
- Start port forwarding for deployment
```
ssh -NfL 9091:localhost:9091 xilinx@137.132.86.241
ssh -NfL 9091:localhost:9091 zenghao@sunfire.comp.nus.edu.sg
```