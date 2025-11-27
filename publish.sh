#!/bin/bash
# Publish Tensora to PyPI
# Usage: ./publish.sh [test|prod]

set -e

TARGET=${1:-test}

echo "=================================================="
echo "  Tensora PyPI Publishing Script"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if twine is installed
if ! command -v twine &> /dev/null; then
    echo -e "${RED}Error: twine is not installed${NC}"
    echo "Install with: pip install twine"
    exit 1
fi

# Get version
VERSION=$(grep -oP "__version__ = ['\"]([^'\"]+)" tensora/__init__.py | grep -oP "[0-9.]+")
echo -e "${GREEN}Building version: $VERSION${NC}"
echo ""

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info
echo ""

# Build source distribution
echo "📦 Building source distribution..."
python setup.py sdist
echo ""

# Check the distribution
echo "🔍 Checking distribution..."
twine check dist/*
if [ $? -ne 0 ]; then
    echo -e "${RED}Distribution check failed!${NC}"
    exit 1
fi
echo ""

# Upload based on target
if [ "$TARGET" == "test" ]; then
    echo -e "${YELLOW}📤 Uploading to TestPyPI...${NC}"
    echo "You can test install with:"
    echo "  pip install --index-url https://test.pypi.org/simple/ tensora"
    echo ""
    twine upload --repository testpypi dist/*
elif [ "$TARGET" == "prod" ]; then
    echo -e "${YELLOW}📤 Uploading to PyPI (PRODUCTION)...${NC}"
    read -p "Are you sure you want to publish to production PyPI? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
    twine upload dist/*
else
    echo -e "${RED}Unknown target: $TARGET${NC}"
    echo "Usage: $0 [test|prod]"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Done!${NC}"
echo ""

if [ "$TARGET" == "prod" ]; then
    echo "Users can now install with:"
    echo "  pip install tensora"
    echo ""
    echo "Don't forget to:"
    echo "  1. Create a git tag: git tag v$VERSION && git push --tags"
    echo "  2. Create a GitHub release"
fi
