"""
Поиск и сбор списков блогерских каналов через API сервисов.

Источники:
- TGStat API (https://tgstat.ru/api) — каталог Telegram-каналов
- VK API groups.search — поиск сообществ ВКонтакте
"""

import time
from typing import List, Dict, Optional
import pandas as pd
import requests


# ============================================================================
# 1. TGStat — поиск Telegram-каналов
# ============================================================================
# Бесплатный токен: регистрация на https://tgstat.ru/api
# Лимит: ~100 запросов/день на бесплатном тарифе

TGSTAT_BASE = 'https://api.tgstat.ru'


class TGStatFinder:
    """
    Поиск Telegram-каналов через TGStat API.

    Нужен токен — получается бесплатно на https://tgstat.ru/api
    """

    # Категории TGStat (основные для блогеров)
    BLOGGER_CATEGORIES = {
        'lifestyle':   'Блоги / Лайфстайл',
        'beauty':      'Красота и здоровье',
        'travel':      'Путешествия',
        'food':        'Еда и кулинария',
        'sport':       'Спорт',
        'fashion':     'Мода',
        'tech':        'Технологии',
        'business':    'Бизнес и стартапы',
        'marketing':   'Маркетинг и SMM',
        'auto':        'Автомобили',
        'humor':       'Юмор и развлечения',
        'education':   'Образование и наука',
        'psychology':  'Психология',
        'family':      'Семья и дети',
        'culture':     'Культура и искусство',
        'politics':    'Политика',
        'economics':   'Экономика',
        'crypto':      'Криптовалюты',
    }

    def __init__(self, token: str):
        self.token = token

    def _request(self, endpoint: str, params: Dict) -> Dict:
        """Запрос к TGStat API."""
        params['token'] = self.token
        url = f'{TGSTAT_BASE}/{endpoint}'
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get('status') != 'ok':
            raise RuntimeError(f'TGStat API error: {data.get("error", "unknown")}')
        return data.get('response', {})

    def search_channels(self, category: str, limit: int = 50,
                        offset: int = 0, language: str = 'ru',
                        country: str = 'ru', min_subscribers: int = 1000,
                        max_subscribers: int = 10_000_000) -> List[Dict]:
        """
        Поиск каналов по категории.

        Параметры
        ---------
        category : str — ключ из BLOGGER_CATEGORIES
        limit : int — макс. каналов за запрос
        language : str — 'ru' русскоязычные
        country : str — 'ru' Россия
        min_subscribers : int — минимальное число подписчиков
        """
        params = {
            'category': category,
            'language': language,
            'country': country,
            'limit': min(limit, 50),  # API limit
            'offset': offset,
            'min_subscribers': min_subscribers,
            'max_subscribers': max_subscribers,
            'extended': 1,  # подробная информация
        }
        resp = self._request('channels/search', params)
        items = resp.get('items', [])

        results = []
        for ch in items:
            results.append({
                'platform': 'Telegram',
                'username': f"@{ch.get('username', '')}" if ch.get('username') else '',
                'title': ch.get('title', ''),
                'followers_count': ch.get('participants_count', 0) or ch.get('subscribers', 0),
                'category': category,
                'category_name': self.BLOGGER_CATEGORIES.get(category, category),
                'avg_views': ch.get('avg_views', 0),
                'avg_forwards': ch.get('avg_forwards', 0),
                'er_views': ch.get('err', 0) or ch.get('engagement_rate', 0),
                'description': (ch.get('about', '') or '')[:200],
            })
        return results

    def collect_all_categories(self, categories: List[str] = None,
                               per_category: int = 50,
                               min_subscribers: int = 3000,
                               delay: float = 1.0) -> pd.DataFrame:
        """
        Сбор каналов по всем заданным категориям.

        Возвращает DataFrame со списком каналов (без постов).
        """
        if categories is None:
            # Только блогерские категории, не новостные
            categories = ['lifestyle', 'beauty', 'travel', 'food', 'sport',
                          'fashion', 'tech', 'business', 'marketing', 'auto',
                          'humor', 'education', 'psychology', 'family']

        all_channels = []
        for cat in categories:
            try:
                channels = self.search_channels(
                    category=cat,
                    limit=per_category,
                    min_subscribers=min_subscribers,
                )
                all_channels.extend(channels)
                print(f'[TGStat] {cat}: найдено {len(channels)} каналов')
                time.sleep(delay)
            except Exception as e:
                print(f'[ERR] {cat}: {e}')

        df = pd.DataFrame(all_channels)
        # Убираем дубликаты по username
        df = df.drop_duplicates(subset=['username'])
        print(f'\nВсего найдено TG-каналов: {len(df)}')
        return df


# ============================================================================
# 2. VK API — поиск сообществ блогеров
# ============================================================================

class VKFinder:
    """
    Поиск VK-сообществ через VK API groups.search.
    """

    # Поисковые запросы для поиска блогеров по тематикам
    SEARCH_QUERIES = {
        'Лайфстайл':   ['блог', 'лайфстайл', 'блогер', 'личный блог'],
        'Красота':     ['beauty', 'бьюти', 'косметика', 'уход', 'макияж'],
        'Путешествия': ['travel', 'путешествия', 'тревел', 'travel blog'],
        'Еда':         ['food', 'рецепты', 'кулинария', 'еда', 'food blog'],
        'Спорт':       ['фитнес', 'спорт', 'тренировки', 'зож', 'fitness'],
        'Мода':        ['fashion', 'стиль', 'мода', 'одежда'],
        'Технологии':  ['tech', 'техно', 'гаджеты', 'обзоры', 'technology'],
        'Бизнес':      ['бизнес', 'предприниматель', 'business', 'стартап'],
        'Маркетинг':   ['smm', 'маркетинг', 'digital', 'продвижение'],
        'Авто':        ['авто', 'автомобили', 'auto', 'drive'],
        'Юмор':        ['юмор', 'приколы', 'мемы', 'смех'],
        'Психология':  ['психология', 'отношения', 'саморазвитие'],
        'Семья':       ['дети', 'мама', 'семья', 'родители', 'mama blog'],
    }

    def __init__(self, access_token: str, api_version: str = '5.199'):
        self.token = access_token
        self.version = api_version

    def _vk_api(self, method: str, params: Dict) -> Dict:
        """Запрос к VK API."""
        params['access_token'] = self.token
        params['v'] = self.version
        url = f'https://api.vk.com/method/{method}'
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
        if 'error' in data:
            raise RuntimeError(f"VK API: {data['error'].get('error_msg', 'unknown')}")
        return data.get('response', {})

    def search_groups(self, query: str, count: int = 50,
                      offset: int = 0, group_type: str = '') -> List[Dict]:
        """
        Поиск сообществ по ключевому запросу.

        group_type: '' (все), 'group', 'page', 'event'
        """
        params = {
            'q': query,
            'sort': 0,
            'count': min(count, 200),
            'offset': offset,
            'country_id': 1,
        }
        if group_type:
            params['type'] = group_type
        resp = self._vk_api('groups.search', params)
        items = resp.get('items', [])

        results = []
        for g in items:
            results.append({
                'platform': 'VK',
                'username': g.get('screen_name', ''),
                'title': g.get('name', ''),
                'followers_count': g.get('members_count', 0),
                'query': query,
                'is_closed': g.get('is_closed', 0),
                'description': (g.get('description', '') or '')[:200],
            })
        return results

    def collect_all_queries(self, queries: Dict[str, List[str]] = None,
                            per_query: int = 30,
                            delay: float = 0.35) -> pd.DataFrame:
        """
        Сбор сообществ по всем поисковым запросам.
        """
        if queries is None:
            queries = self.SEARCH_QUERIES

        all_groups = []
        for category, keywords in queries.items():
            for kw in keywords:
                try:
                    groups = self.search_groups(query=kw, count=per_query)
                    for g in groups:
                        g['category'] = category
                    all_groups.extend(groups)
                    print(f'[VK] "{kw}": найдено {len(groups)}')
                    time.sleep(delay)
                except Exception as e:
                    print(f'[ERR] "{kw}": {e}')

        df = pd.DataFrame(all_groups)
        if df.empty:
            print('\nVK-сообществ не найдено')
            return df
        df = df.drop_duplicates(subset=['username'])
        if 'is_closed' in df.columns:
            df = df[df['is_closed'] == 0]
        print(f'\nВсего найдено VK-сообществ: {len(df)}')
        return df


# ============================================================================
# Резервный метод: ручные списки популярных российских блогеров
# ============================================================================
# Если TGStat/VK API недоступны, используй готовые списки (подставь свои username'ы)

MANUAL_TG_BLOGGERS = [
    # Топ российских блогеров (проверено — каналы существуют)
    # Лайфстайл
    '@lerchek', '@kategordon', '@belonika', '@primemeat',
    '@nastyapom', '@sonchicc', '@stolyarovalive', '@yana_rudkovskaya',
    '@mashulik', '@tanyastarikova',
    # IT / Техно
    '@wylsacom', '@bobuk', '@stalker_blog', '@eldarmurtazin',
    '@biggeek', '@michael_naki',
    # Бизнес
    '@shabutdinov', '@krylov', '@amigo987',
    # Маркетинг
    '@igoryak', '@shardakov', '@nikkagent',
    # Путешествия
    '@varlamov',
]

MANUAL_VK_BLOGGERS = [
    'lerchek', 'kategordon', 'belonika', 'wylsacom',
    'eldarmurtazin', 'varlamov', 'shabutdinov', 'krylov',
    'igoryak', 'shardakov', 'amigo987',
]
