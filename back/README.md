# Backend

This API has been deployed on heroku :
https://proj8backend-5060a747d1a0.herokuapp.com/

## Model 

The model used for the prediction is a lightgbm trained on 307,511 clients and 62 features characterizing client's ability to repay a loan (info from previous loan, personnal info, external sources ...) 



## API content

This API is created with the FASTAPI framework in python and use two classes (one inuput and one output):

* The first class : Client_data contains the 62 features defining the client 
* The second class : Client_Target contains the probability and the default loan prediction (0 or 1)\


The main function is a POST request called /predict using the class Client_data as an input template and Client_Target as an output template.
 
