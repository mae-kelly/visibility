# Autonomous Visibility Platform - Frontend

## Quick Start

### Option 1: Local Development
```bash
cd frontend
./start.sh
```

### Option 2: Manual Setup
```bash
cd frontend
npm install
npm run dev
```

### Option 3: Docker
```bash
cd frontend
docker build -t visibility-frontend .
docker run -p 3000:3000 visibility-frontend
```

## Features

- **Real-time Dashboard** with live metrics
- **Asset Management** with search and filtering
- **Visibility Gap Analysis** with actionable insights
- **AI Model Monitoring** with performance metrics
- **Responsive Design** works on all devices
- **Modern UI** with Tailwind CSS

## Environment Variables

Create `.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Testing

Once running on http://localhost:3000:

- Dashboard: http://localhost:3000/
- Assets: http://localhost:3000/assets
- Gaps: http://localhost:3000/gaps

## API Integration

The frontend automatically connects to the backend API running on port 8000.
Make sure your backend is running first!
