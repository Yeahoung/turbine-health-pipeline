.PHONY: start-all run-app build-presentation test-rl docker-up docker-build

start-all:
	@chmod +x start_all.sh
	@./start_all.sh

run-app:
	streamlit run app.py

build-presentation:
	cd presentation && npm install && npm run build

test-rl:
	cd rl_agent && python train.py

# Full-stack Docker: Presentation + App in one container
docker-build:
	docker compose build

docker-up:
	docker compose up --build
