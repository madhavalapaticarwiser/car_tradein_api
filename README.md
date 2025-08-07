# car_tradein_api
FastAPI + Docker container for car tradein price estimates

# 1. (Optional)
git clone https://github.com/madhavalapaticarwiser/car-tradein-api.git
cd car-tradein-api

# 2. Pull the pre-built image
docker pull madhavalapati/car-tradein-api:1.0.1

# 3. Run the container
docker run --rm -p 8080:8080 madhavalapati/car-tradein-api:1.0.1

# 1.1. Build the Docker image (Optional)
docker build -t car-tradein-api .

# 1.2. Run the container (Optional)
docker run --rm -p 8080:8080 car-tradein-api

# 4. Verify in browser or via cURL
#    Browser: http://localhost:8080/docs
#    cURL:
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
        "year": 2024,
        "mileage": 10000,
        "make": "Toyota",
        "model": "Camron",
        "trim": "XLE Auto",
        "interior": "great",
        "exterior": "great",
        "mechanical": "great",
        "line": "Economy",
        "drivetrain": "FWD",
        "transmission": "Automatic"
      }'

Should return
{
  "success": true,
  "price": 24207.19921875,
  "message": null
}
