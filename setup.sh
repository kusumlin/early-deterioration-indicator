#!/bin/bash
# EDI — setup virtual environment and install dependencies

set -e

VENV_DIR=".venv"

echo "Creating virtual environment in $VENV_DIR..."
python3 -m venv "$VENV_DIR"

echo "Activating venv and installing dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "Setup complete."
echo ""
echo "To activate the venv:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then run the project:"
echo "  python pipeline.py          # ETL + DB + report"
echo "  streamlit run dashboard.py  # interactive dashboard"
echo "  python main.py              # ML demo + plots"
