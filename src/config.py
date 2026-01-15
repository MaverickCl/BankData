"""
Configuración del Análisis de Marketing Bancario
================================================
Este módulo contiene todas las configuraciones necesarias para hacer
el código generalizable y aplicable a nuevos datasets.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ColumnMapping:
    """Mapeo de nombres de columnas esperadas en el dataset."""
    target: str = "y"  # Variable objetivo
    contact: str = "contact"  # Canal de contacto
    month: str = "month"  # Mes
    day: str = "day"  # Día del mes
    age: str = "age"  # Edad
    balance: str = "balance"  # Balance bancario
    duration: str = "duration"  # Duración del contacto
    campaign: str = "campaign"  # Número de contactos
    housing: str = "housing"  # Préstamo hipotecario
    loan: str = "loan"  # Préstamo personal
    job: str = "job"  # Ocupación


@dataclass
class ValueMapping:
    """Mapeo de valores categóricos esperados."""
    # Valores de la variable objetivo
    target_positive: str = "yes"
    target_negative: str = "no"
    
    # Valores de préstamos
    has_loan: str = "yes"
    no_loan: str = "no"
    
    # Canales de contacto
    contact_cellular: str = "cellular"
    contact_telephone: str = "telephone"
    
    # Meses (en inglés, se traducirán)
    months_order: List[str] = None
    
    def __post_init__(self):
        if self.months_order is None:
            self.months_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


@dataclass
class GenerationConfig:
    """Configuración para clasificación por generaciones."""
    reference_year: int = 2012  # Año de referencia de los datos
    
    # Rangos de nacimiento por generación
    silent_gen_start: int = 1928
    silent_gen_end: int = 1945
    baby_boomers_start: int = 1946
    baby_boomers_end: int = 1964
    gen_x_start: int = 1965
    gen_x_end: int = 1980
    millennials_start: int = 1981
    millennials_end: int = 1996
    
    def get_age_ranges(self) -> Dict[str, tuple]:
        """Retorna los rangos de edad para cada generación basado en el año de referencia."""
        return {
            'Silent Generation': (
                self.reference_year - self.silent_gen_end,
                self.reference_year - self.silent_gen_start
            ),
            'Baby Boomers': (
                self.reference_year - self.baby_boomers_end,
                self.reference_year - self.baby_boomers_start
            ),
            'Gen X': (
                self.reference_year - self.gen_x_end,
                self.reference_year - self.gen_x_start
            ),
            'Millennials': (
                self.reference_year - self.millennials_end,
                self.reference_year - self.millennials_start
            )
        }


@dataclass
class AnalysisConfig:
    """Configuración general del análisis."""
    # Rutas
    output_dir: Path = Path("reports/figures")
    
    # Configuración de gráficos
    figure_dpi: int = 300
    figure_format: str = "png"
    
    # Configuración de análisis
    outlier_percentile: float = 0.95  # Percentil para filtrar outliers
    
    # Traducciones (para hacer el código más flexible)
    translations: Dict[str, Dict[str, str]] = None
    
    def __post_init__(self):
        if self.translations is None:
            self.translations = {
                'job': {
                    'management': 'Gerencia',
                    'blue-collar': 'Obrero',
                    'technician': 'Tecnico',
                    'admin.': 'Administrativo',
                    'services': 'Servicios',
                    'retired': 'Jubilado',
                    'self-employed': 'Autonomo',
                    'entrepreneur': 'Empresario',
                    'unemployed': 'Desempleado',
                    'housemaid': 'Empleado domestico',
                    'student': 'Estudiante',
                },
                'months': {
                    'jan': 'Ene', 'feb': 'Feb', 'mar': 'Mar', 'apr': 'Abr',
                    'may': 'May', 'jun': 'Jun', 'jul': 'Jul', 'aug': 'Ago',
                    'sep': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'dec': 'Dic'
                }
            }


# Instancia global de configuración (puede ser sobrescrita)
DEFAULT_COLUMNS = ColumnMapping()
DEFAULT_VALUES = ValueMapping()
DEFAULT_GENERATIONS = GenerationConfig()
DEFAULT_ANALYSIS = AnalysisConfig()
