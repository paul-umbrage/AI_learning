#!/bin/bash
# Quick script to update database password in .env

echo "Please enter your PostgreSQL password for user 'postgres':"
read -s DB_PASS

# Update .env file
sed -i.bak "s/^DB_PASSWORD=.*/DB_PASSWORD=$DB_PASS/" .env

echo "Database password updated in .env"
echo "Testing connection..."
source venv/bin/activate
python3 -c "from database import get_db_connection; conn = get_db_connection(); print('✓ Connection successful!'); conn.close()" 2>&1
