#!/bin/bash

# example of using arguments to a script
rq worker --url redis://redis &
uvicorn main:app --host 0.0.0.0 --port 5000