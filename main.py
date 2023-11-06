import base64
import datetime
import io
import json
import os
from io import BytesIO
import boto3

import numpy as np
import requests
from PIL import Image
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext

import logging

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.dispatcher.filters.state import StatesGroup, State

from swaper import FaceSwaper

logging.basicConfig(level=logging.INFO)

from env import BOT_TOKEN
from phrases import *
from bd import get_language, read_user, create_user, setlanguage


inkb = InlineKeyboardMarkup(row_width = 2).add(InlineKeyboardButton(text='russian', callback_data='rus'),
                                               InlineKeyboardButton(text='english', callback_data='eng'))



class SwaperState(StatesGroup):
    Q1 = State()
    Q2 = State()
    Q3 = State()

    Q_language = State()
    Q_feedback = State()

history_img ={}


#@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    user_id = message.from_user.id
    language = get_language(user_id)
    #await message.reply(phrase_start[language])
    await message.answer(phrase_start['eng'], reply_markup=inkb)

    if read_user(user_id) == False:
        create_user(user_id, None, None)

    await SwaperState.Q1.set()

    #await message.reply('&#129327', parse_mode='HTML')

async def change_lang(callback: types.CallbackQuery):
    user_id = callback.message.chat.id
    #print(callback.message.text)
    language = callback.values['data']

    setlanguage(user_id, str(language))
    await callback.message.answer(phrase_language[str(language)])
    #await callback.message.answer('ok')

#@dp.message_handler(commands=['find'])
async def start_swap(message: types.Message, state = FSMContext):
    user_id = message.from_user.id
    await message.answer(phrase_swap_start['eng'], parse_mode='HTML')
    await SwaperState.Q1.set()


#@dp.message_handler(content_types=['photo'], state= SwaperState.Q1)
async def save_im1(message: types.Message, state = FSMContext):
    user_id = message.from_user.id


    file_id = message.photo[-1].file_id
    file_info = await bot.get_file(file_id)
    history_img[user_id] = await file_info.get_url()
    await message.answer(phrase_second_photo['eng'])
    await SwaperState.Q2.set()


#@dp.message_handler(content_types=['photo'], state=SwaperState.Q2)
async def save_im2(message, state = FSMContext):
    user_id = message.from_user.id

    if False:
        #file_id1 = message.photo[-1].file_id
        file_id1 = message.photo[-2].file_unique_id
        file_info = await bot.get_file(file_id1)
        f = BytesIO()
        downloaded_file = await bot.download_file(file_info.file_path, destination=f)
        img1 = Image.open(BytesIO(initial_bytes=f.getvalue()))

        #file_id = message.photo[-1].file_id
        file_id = message.photo[-1].file_unique_id
        file_info = await bot.get_file(file_id)
        f = BytesIO()
        downloaded_file = await bot.download_file(file_info.file_path, destination=f)
        img2 = Image.open(BytesIO(initial_bytes=f.getvalue()))

    img_url = history_img[user_id]
    response = requests.get(img_url, timeout=5)
    img = Image.open(BytesIO(response.content))
    img1 = np.array(img)

    img_url = await message.photo[-1].get_url()
    response = requests.get(img_url, timeout=5)
    img = Image.open(BytesIO(response.content))
    img2 = np.array(img)

    await SwaperState.Q1.set()
    await message.answer(phrase_proccesing_photo['eng'])
    await swap(message, np.array(img1), np.array(img2))



async def swap(message, img1, img2):
    user_id = message.from_user.id

    fs = FaceSwaper()
    img_result = fs.swap(img1, img2)
    #history_dict.pop(user_id)  # delete
    if type(img_result) == type(list()):
        #await message.answer(f"Not founded face on {img_result[1]} photo")
        await message.answer(phrase_not_face['eng'])
        return

    image = Image.fromarray(img_result)

    bio = BytesIO()
    bio.name = 'image.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)


    await bot.send_photo(user_id, photo = bio)


class SwaperState(StatesGroup):
    Q1 = State()
    Q2 = State()
    Q3 = State()

    Q_language = State()
    Q_feedback = State()


if __name__ == '__main__':
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher(bot, storage=MemoryStorage())

    dp.register_message_handler(start, commands=['start'])
    dp.register_message_handler(start_swap, commands=['swap'])
    dp.register_message_handler(save_im1, content_types=['photo'], state= SwaperState.Q1)
    dp.register_message_handler(save_im2, content_types=['photo'], state=SwaperState.Q2)
    dp.register_callback_query_handler(change_lang, text=['rus', 'eng'])

    bot.get_updates(offset=-1)
    Bot.set_current(dp.bot)
    Dispatcher.set_current(dp)

    executor.start_polling(dp, skip_updates=True)
