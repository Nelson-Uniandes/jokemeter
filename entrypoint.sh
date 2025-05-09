#!/bin/bash
set -e

echo "📦 Ejecutando dvc pull para restaurar modelos..."
dvc pull

echo "🚀 Iniciando la API..."
exec "$@"