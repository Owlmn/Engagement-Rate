"""Сборщик Telegram-каналов через Telethon MTProto API."""

import asyncio, re, os
from datetime import datetime, timezone
from typing import List, Dict, Optional
import pandas as pd
from telethon import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.types import Message, User
from telethon.errors import FloodWaitError, ChannelPrivateError
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

class TelegramCollector:
    def __init__(self, api_id: int, api_hash: str, session: str = 'tg'):
        self.client = TelegramClient(session, api_id, api_hash)

    async def start(self):
        await self.client.start()
        me = await self.client.get_me()
        print(f'[TG] @{me.username or me.first_name}')
        return self

    async def stop(self):
        await self.client.disconnect()

    async def _channel_info(self, username: str) -> Dict:
        try:
            e = await self.client.get_entity(username)
            full = await self.client(GetFullChannelRequest(username))
            return {
                'channel_username': username,
                'channel_title': getattr(e, 'title', username),
                'channel_id': e.id,
                'followers_count': full.full_chat.participants_count or 0,
            }
        except Exception as ex:
            print(f'[WARN] {username}: {ex}')
            return {'channel_username': username, 'channel_title': username,
                    'channel_id': None, 'followers_count': 0}

    def _parse(self, msg: Message, info: Dict) -> Dict:
        text = msg.message or ''
        links = re.findall(r'https?://\S+', text)
        hashtags = re.findall(r'#\w+', text)
        mentions = re.findall(r'@\w+', text)
        media = 'none'
        if msg.media:
            if getattr(msg, 'photo', None): media = 'photo'
            elif getattr(msg, 'video', None): media = 'video'
            elif getattr(msg, 'document', None):
                doc = msg.document
                mime = getattr(doc, 'mime_type', '') if doc else ''
                media = 'video_file' if 'video' in mime else ('gif_image' if 'gif' in mime else 'document')
            else: media = 'other'
        reactions = sum(r.count for r in msg.reactions.results) if hasattr(msg, 'reactions') and msg.reactions and msg.reactions.results else 0
        return {
            'platform': 'Telegram', 'channel_username': info['channel_username'],
            'channel_title': info['channel_title'], 'followers_count': info['followers_count'],
            'post_id': msg.id, 'post_text': text, 'text_length': len(text) if text else 0,
            'has_media': bool(msg.media), 'media_type': media,
            'has_photo': media == 'photo', 'has_video': media in ('video', 'video_file'),
            'has_document': media == 'document',
            'has_link': len(links) > 0, 'n_links_raw': len(links),
            'has_hashtag': len(hashtags) > 0, 'n_hashtags_raw': len(hashtags),
            'n_mentions_raw': len(mentions),
            'views_count': getattr(msg, 'views', 0) or 0,
            'forwards_count': getattr(msg, 'forwards', 0) or 0,
            'replies_count': msg.replies.replies if msg.replies else 0,
            'reactions_count': reactions,
            'likes_count': 0, 'comments_count': 0,
            'is_advert': is_advert(text), 'country': '', 'city': '',
            'published_at': msg.date.replace(tzinfo=timezone.utc).isoformat() if msg.date else None,
            'collected_at': datetime.now(timezone.utc).isoformat(),
        }

    async def _fetch(self, username: str, limit: int, delay: float) -> List[Dict]:
        posts = []
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                entity = await self.client.get_entity(username)
                if isinstance(entity, User):
                    print(f'[SKIP] {username}: user profile, not a channel')
                    return posts
                info = await self._channel_info(username)
                count = 0
                async for msg in self.client.iter_messages(entity, limit=limit):
                    if msg.message or msg.text:
                        posts.append(self._parse(msg, info))
                        count += 1
                    if delay > 0 and count % 10 == 0:
                        await asyncio.sleep(delay * 0.35)
                print(f'[OK] {username}: {count}')
                return posts
            except FloodWaitError as e:
                wait_time = e.seconds
                print(f'[FLOOD] {username}: жду {wait_time}s (попытка {retry_count + 1}/{max_retries})')
                # Ждём столько, сколько требует Telegram (не ограничиваем)
                await asyncio.sleep(wait_time)
                retry_count += 1
            except ChannelPrivateError:
                print(f'[SKIP] {username}: private')
                return posts
            except Exception as e:
                print(f'[ERR] {username}: {e}')
                if retry_count < max_retries - 1:
                    backoff = 2 ** retry_count  # Экспоненциальный backoff
                    print(f'[RETRY] жду {backoff}s перед повтором...')
                    await asyncio.sleep(backoff)
                    retry_count += 1
                else:
                    return posts
        return posts

    async def collect(self, channels: List[str], output: str,
                      limit: int = 200, delay: float = 0.5,
                      batch: int = 20, rest: int = 600) -> pd.DataFrame:
        
        if os.path.exists(output):
            try:
                old = pd.read_csv(output)
                done = set(old['channel_username'].unique())
                all_posts = old.to_dict('records')
                print(f'[RESUME] {len(old)} постов из {len(done)} каналов')
            except Exception:
                done, all_posts = set(), []
        else:
            done, all_posts = set(), []

        remaining = [c for c in channels if c not in done]
        print(f'Осталось: {len(remaining)}/{len(channels)}')

        for i, ch in enumerate(tqdm(remaining, desc='TG')):
            posts = await self._fetch(ch, limit, delay)
            all_posts.extend(posts)
            done.add(ch)
            pd.DataFrame(all_posts).to_csv(output, index=False)
            await asyncio.sleep(delay * 2)  # Пауза между каналами
            if (i + 1) % batch == 0:
                print(f'\n[PAUSE] {i+1}/{len(remaining)}, {rest}s...')
                await asyncio.sleep(rest)

        df = pd.DataFrame(all_posts)
        df.to_csv(output, index=False)
        print(f'\n=== TG: {len(df)} постов из {len(done)} каналов ===')
        return df
