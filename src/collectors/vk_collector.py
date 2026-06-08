"""Сборщик VK-сообществ через VK API."""

import re, time, os
from datetime import datetime, timezone
from typing import List, Dict
import pandas as pd
import vk_api
from vk_api.exceptions import ApiError
from tqdm import tqdm

ADVERT_PATTERNS = [
    r'#реклама', r'#спонсор', r'#партнерский', r'#промо',
    r'#сотрудничество', r'#интеграция', r'#ad', r'#sponsored',
    r'erid:', r'токен:', r'erid\s*:',
    r'реклама\s*[:.]', r'рекламный пост', r'на правах рекламы',
    r'партнёрский материал', r'спонсорский пост',
    r'ооо\s+', r'ип\s+',
]

def is_advert(text: str) -> bool:
    if not text: return False
    t = text.lower()
    for p in ADVERT_PATTERNS:
        if re.search(p, t): return True
    return False

class VKCollector:
    def __init__(self, token: str, version: str = '5.199'):
        self.vk = vk_api.VkApi(token=token, api_version=version).get_api()

    def _group_info(self, gid: str) -> Dict:
        try:
            resp = self.vk.groups.getById(group_id=gid, fields='members_count,country,city')
            if resp:
                g = resp[0]
                return {
                    'group_id': gid, 'group_name': g.get('name', gid),
                    'group_screen_name': g.get('screen_name', gid),
                    'followers_count': g.get('members_count', 0),
                    'group_id_internal': g.get('id', 0),
                    'country': (g.get('country', {}) or {}).get('title', ''),
                    'city': (g.get('city', {}) or {}).get('title', ''),
                }
        except ApiError as e:
            print(f'[WARN] {gid}: {e}')
        return {'group_id': gid, 'group_name': gid, 'group_screen_name': gid,
                'followers_count': 0, 'group_id_internal': 0, 'country': '', 'city': ''}

    def _parse(self, item: Dict, info: Dict) -> Dict:
        text = item.get('text', '')
        att = item.get('attachments', [])
        types = [a.get('type') for a in att]
        links = re.findall(r'https?://\S+', text)
        hashtags = re.findall(r'#\w+', text)
        mentions = re.findall(r'@\w+', text)
        views = item.get('views', {})
        likes = item.get('likes', {})
        reposts = item.get('reposts', {})
        comments = item.get('comments', {})
        ts = item.get('date', 0)
        return {
            'platform': 'VK', 'channel_username': info.get('group_screen_name', info['group_id']),
            'channel_title': info.get('group_name', info['group_id']),
            'followers_count': info.get('followers_count', 0),
            'post_id': item.get('id', 0), 'post_text': text,
            'text_length': len(text) if text else 0,
            'has_media': bool(att), 'media_type': types[0] if types else 'none',
            'has_photo': 'photo' in types, 'has_video': 'video' in types,
            'has_document': 'doc' in types, 'has_poll': 'poll' in types,
            'has_link': 'link' in types or len(links) > 0,
            'n_links_raw': len(links), 'has_hashtag': len(hashtags) > 0,
            'n_hashtags_raw': len(hashtags), 'n_mentions_raw': len(mentions),
            'views_count': views.get('count', 0) if isinstance(views, dict) else 0,
            'likes_count': likes.get('count', 0) if isinstance(likes, dict) else 0,
            'forwards_count': reposts.get('count', 0) if isinstance(reposts, dict) else 0,
            'comments_count': comments.get('count', 0) if isinstance(comments, dict) else 0,
            'reactions_count': 0, 'replies_count': 0,
            'is_advert': is_advert(text) or bool(item.get('marked_as_ads', 0)),
            'country': info.get('country', ''), 'city': info.get('city', ''),
            'published_at': datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
            'collected_at': datetime.now(timezone.utc).isoformat(),
        }

    def _fetch(self, gid: str, limit: int, delay: float) -> List[Dict]:
        posts, offset = [], 0
        bs = min(100, limit)
        try:
            info = self._group_info(gid)
        except Exception:
            info = {'group_id': gid, 'group_name': gid, 'group_screen_name': gid,
                    'followers_count': 0, 'group_id_internal': 0, 'country': '', 'city': ''}
        oid = -abs(info['group_id_internal']) if info['group_id_internal'] else None
        
        retry_count = 0
        max_retries = 3
        
        while len(posts) < limit:
            try:
                resp = self.vk.wall.get(owner_id=oid, domain=gid if not oid else None,
                                        count=bs, offset=offset, filter='all', extended=0)
                items = resp.get('items', [])
                if not items: break
                for item in items:
                    if len(posts) >= limit: break
                    posts.append(self._parse(item, info))
                offset += len(items)
                time.sleep(delay)
                retry_count = 0  # Обнулить счётчик при успехе
                if offset >= 2500: break
            except ApiError as e:
                err = str(e)
                # Невалидный domain/screen_name — сразу скипаем
                if '100' in err:
                    print(f'[SKIP] {gid}: invalid domain')
                    break
                print(f'[VK] {gid}: {err}')
                
                is_flood = 'Flood control' in err or '9' in err or '6' in err
                if is_flood and retry_count < max_retries:
                    wait_time = 180 * (2 ** retry_count)
                    print(f'[FLOOD_VK] ждём {wait_time}s (попытка {retry_count + 1}/{max_retries})')
                    time.sleep(wait_time)
                    retry_count += 1
                else:
                    time.sleep(5)
                    break
        print(f'[OK] {gid}: {len(posts)}')
        return posts

    def collect(self, groups: List[str], output: str,
                limit: int = 200, delay: float = 1.0,
                batch: int = 10, rest: int = 300) -> pd.DataFrame:
        
        if os.path.exists(output):
            try:
                old = pd.read_csv(output)
                done = set(old['channel_username'].unique())
                all_posts = old.to_dict('records')
                print(f'[RESUME] {len(old)} постов из {len(done)} групп')
            except Exception:
                done, all_posts = set(), []
        else:
            done, all_posts = set(), []

        remaining = [g for g in groups if g not in done]
        print(f'Осталось: {len(remaining)}/{len(groups)}')

        for i, gid in enumerate(tqdm(remaining, desc='VK')):
            posts = self._fetch(gid, limit, delay)
            all_posts.extend(posts)
            done.add(gid)
            pd.DataFrame(all_posts).to_csv(output, index=False)
            time.sleep(delay * 2)
            if (i + 1) % batch == 0:
                print(f'\n[PAUSE] {i+1}/{len(remaining)}, {rest}s...')
                time.sleep(rest)

        df = pd.DataFrame(all_posts)
        df.to_csv(output, index=False)
        print(f'\n=== VK: {len(df)} постов из {len(done)} групп ===')
        return df
