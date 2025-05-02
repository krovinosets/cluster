import os
from argparse import ArgumentParser

from dotenv import load_dotenv

from src.parser import Parser
from src.cluster import get_vk_posts_texts, get_vectors, number_of_clusters, kmeans_clustering, \
    get_cluster_keywords, make_2d_plot, make_3d_plot

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
            texts, group_ids = get_vk_posts_texts()
            vectors = get_vectors(texts)
            num = number_of_clusters(vectors)
            labels, centers = kmeans_clustering(vectors, num, group_ids)
            get_cluster_keywords(texts, labels)
            make_2d_plot(vectors, texts, labels, group_ids)
            make_3d_plot(vectors, texts, labels, group_ids)
