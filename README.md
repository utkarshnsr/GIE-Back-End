## Running stand-alone

`python data_extractor.py`

## Running the server

Happens in 2 phases - first run the angular front-end server and then run the backend server using uvicorn
1. download the GIE-front-end repo code
2. run `ng serve --open`
3. run `uvicorn server:app --reload`

_there are many additional files which can be ignored or used for trial and error like `line_trial.py`_
