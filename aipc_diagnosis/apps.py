from django.apps import AppConfig

class AipcDiagnosisConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'aipc_diagnosis'  

    def ready(self):
        import aipc_diagnosis.signals  

