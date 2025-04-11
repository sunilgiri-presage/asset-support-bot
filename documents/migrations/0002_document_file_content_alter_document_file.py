# Generated by Django 5.1.7 on 2025-03-25 18:24

import documents.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('documents', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='file_content',
            field=models.BinaryField(blank=True, help_text='File content stored directly in the database', null=True),
        ),
        migrations.AlterField(
            model_name='document',
            name='file',
            field=models.FileField(help_text='Document file stored in the database', storage=documents.models.DatabaseStorage(), upload_to='documents/'),
        ),
    ]
