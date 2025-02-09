# Variables
PYTHON = python3.10
POETRY = poetry

# Mostrar ayuda
.PHONY: help
help:
	@echo "Comandos disponibles:"
	@echo "  setup    - Configurar el entorno e instalar dependencias"
	@echo "  serve    - Iniciar el servidor con uvicorn"
	@echo "  test     - Ejecutar pruebas con pytest"
	@echo "  format   - Formatear el código con ruff"
	@echo "  clean    - Limpiar archivos generados"

# Configuración del entorno
.PHONY: setup
setup:
	$(POETRY) env use $(PYTHON)
	$(POETRY) install --no-root

# Iniciar el servidor
.PHONY: serve
serve:
	$(POETRY) run uvicorn api.main:app --reload

# Ejecutar pruebas
.PHONY: test
test:
	$(POETRY) run pytest

# Formatear el código
.PHONY: format
format:
	$(POETRY) run ruff check

# Limpiar archivos generados
.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .venv