import os
from unittest.mock import Mock, patch

from dotenv import load_dotenv


def test_post_slack_with_webhook_url() -> None:
    """Slack webhook経由での通知をテスト"""
    from src.utils.utils import _post_slack_webhook

    webhook_url = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
    channel = "#test"
    username = "test-bot"
    message = "Test message"

    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        status_code = _post_slack_webhook(webhook_url, channel, username, message)

        # リクエストが正しく呼ばれたかチェック
        mock_post.assert_called_once_with(
            webhook_url,
            headers={"Content-Type": "application/json"},
            json={
                "channel": channel,
                "username": username,
                "text": message,
            },
        )
        assert status_code == 200


def test_post_slack_with_token() -> None:
    """Slack API token経由での通知をテスト"""
    from src.utils.utils import _post_slack

    token = "xoxb-test-token"
    channel = "#test"
    username = "test-bot"
    message = "Test message"

    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        status_code = _post_slack(token, channel, username, message)

        # リクエストが正しく呼ばれたかチェック
        mock_post.assert_called_once_with(
            "https://slack.com/api/chat.postMessage",
            headers={"Content-Type": "application/json"},
            params={
                "token": token,
                "channel": channel,
                "text": message,
                "username": username,
            },
        )
        assert status_code == 200


def test_post_slack_webhook_url() -> None:
    from src.utils.utils import _post_slack_webhook

    load_dotenv()

    _post_slack_webhook(
        webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
        channel="#test",
        username="test-bot",
        message="Test message",
    )


def test_post_slack_token() -> None:
    from src.utils.utils import _post_slack

    load_dotenv()

    _post_slack(
        token=os.getenv("SLACK_TOKEN"),
        channel="#test",
        username="test-bot",
        message="Test message",
    )
