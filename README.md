# comms_external
Instrcution:

## Ultra96_Server.py
  The server running on the Ultra96 which in charge of receiving data from different dancers' laptops and pass the data to the ML part. 
  Port number is fixed, which are 9091, 9092, 9093.

  To execute:
  /*py Ultra96_Server.py */
  
## laptop_client.py
  The client running on different dancers' laptops which will transfer the data to Ultra96_Server.
  Port number is fixed, for dancer 0,1,2 are 9091,9092,9093 respectively.
  Password is also fixed, which is 1234123412341234
  
  To execute:
  /*py laptop_client.py */
  
## eval_client.py
  The client sending data back to evaluation server. Running on Ultra96
