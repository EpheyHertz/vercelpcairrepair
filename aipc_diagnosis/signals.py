from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import User
from .models import UserProfile
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.core.cache import cache

from .models import NewsArticle


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()


@receiver([post_save, post_delete], sender=NewsArticle)
def invalidate_news_cache(sender, instance, **kwargs):
    """
    Clear all cached news data when a NewsArticle is created, updated, or deleted.
    """
    # You can use a common key prefix to invalidate related keys
    key_prefix = "tech_news:"
    keys_to_clear = [key for key in cache._cache.keys() if key.startswith(key_prefix)]

    for key in keys_to_clear:
        cache.delete(key)

