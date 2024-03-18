# meeting_summarization_portal/api/serializers.py

from rest_framework import serializers

class ModelLinkSerializer(serializers.Serializer):
    model_link = serializers.CharField(max_length=200)
