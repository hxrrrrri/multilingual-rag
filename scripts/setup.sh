#!/usr/bin/env bash
# One-shot local dev setup
set -e

echo "==> Copying .env"
cp backend/.env.example backend/.env

echo "==> Creating Python venv"
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..

echo "==> Installing frontend dependencies"
cd frontend
npm install
cd ..

echo ""
echo "Setup complete!"
echo "Start services:  docker-compose -f docker/docker-compose.yml up -d"
echo "Start backend:   cd backend && uvicorn app.main:app --reload"
echo "Start frontend:  cd frontend && npm run dev"
