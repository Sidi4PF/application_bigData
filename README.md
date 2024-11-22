### Commands:

#### Docker Image:

docker build -t weather-predictor .

#### Docker Container:	

docker run --rm -it `
  -v "${PWD}/pictures:/app/input" `
  -v "${PWD}/output:/app/output" `
  weather-predictor:latest `
  --input_dir /app/input --output_dir /app/output

### Todos:
- Resilience (able to manage empty/bad data)
- Bonus: Make the python app not predict already predicted images but onlyif asked not to do so
- Bonus: Add automatic packaging using DevopsTool (Travis CI for example)
- Documentation
