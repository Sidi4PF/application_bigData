### Documentation
in folder doc

### Commands:

#### Docker Image:

docker build -t weather-predictor .

#### Docker Container:	

docker run --rm -it `
  -v "${PWD}/pictures:/app/input" `
  -v "${PWD}/output:/app/output" `
  weather-predictor:latest `
  --input_dir /app/input --output_dir /app/output
