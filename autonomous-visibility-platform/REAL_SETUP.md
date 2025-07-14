# Real AO1 Visibility Platform Setup

## 1. Update Database Path
Edit `backend/app/main.py` line 15:
```python
DB_PATH = "/full/path/to/your/actual/database.duckdb"
```

## 2. Start Backend
```bash
cd backend
source venv/bin/activate
cd app  
python main.py
```

## 3. Start Frontend
```bash
cd frontend
npm run dev
```

## 4. Test Your Real Data
Visit: http://localhost:3000

## What You'll See:
- Real asset counts from your Chronicle/Splunk/CrowdStrike/CMDB
- Actual coverage gaps that need fixing
- Actionable recommendations based on your data
- Asset correlation across all your tools

## API Endpoints:
- `/api/v1/coverage/metrics` - Real coverage stats
- `/api/v1/assets/all` - All correlated assets  
- `/api/v1/gaps/list` - Gaps to fix
- `/api/v1/gaps/by-source` - Gaps by tool
