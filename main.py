import os
from argparse import ArgumentParser

from dotenv import load_dotenv

from src.parser import Parser
from src.cluster import get_vk_posts_texts, make_dbscan


load_dotenv()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'mode',
        type=str,
        help='script mode: login, parse or cluster',
        default="parse",
        choices=["login", "parse", "cluster"],
    )

    args = parser.parse_args()
    app_mode: str = args.mode

    match app_mode:
        case "login":
            import webbrowser

            uri = (f"https://oauth.vk.com/authorize?"
                   f"client_id={os.getenv('VK_APP_ID')}&"
                   f"display=page&"
                   f"redirect_uri=https://oauth.vk.com/blank.html&"
                   f"scope=wall,offline&"
                   f"response_type=token&"
                   f"v=5.131")
            webbrowser.open(uri, new=0, autoraise=True)
        case "parse":
            vk_parser = Parser(
                os.getenv("VK_ACCESS_TOKEN")
            )
            posts: list[dict] = vk_parser.get_posts(
                os.getenv("VK_GROUP_ID"),
                int(os.getenv("VK_POSTS_TO_CHECK")),
            )
            vk_parser.save_to_db(posts)
        case "cluster":
            texts: list[str] = get_vk_posts_texts()
            make_dbscan(texts)
