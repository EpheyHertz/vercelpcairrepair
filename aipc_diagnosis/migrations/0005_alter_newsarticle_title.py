# Generated by Django 5.1.1 on 2024-10-22 14:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('aipc_diagnosis', '0004_remove_diagnosis_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='newsarticle',
            name='title',
            field=models.CharField(default='No title', max_length=500, unique=True),
        ),
    ]
