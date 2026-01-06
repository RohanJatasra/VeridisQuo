#!/bin/bash
cd "$(dirname "$0")/../frontend" || exit

echo "📦 Installing dependencies..."
npm install

echo "🚀 Running the application..."
npm run dev