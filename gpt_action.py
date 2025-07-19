from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Nicolacus Maximus Quote giver",
    description="get a real quote said by Nicolacus Maximus himself",
    servers=[{"url": "https://upload-contribution-rx-pays.trycloudflare.com"}],
)


class Quote(BaseModel):
    quote: str = Field(description="The quote that nicolacus Maximus said.")
    year: int = Field(description="the year when Nicolacus Maximus said the quote.")


@app.get(
    "/quote",
    summary="Returns a random quote by Nicolacus Maximus",
    description="Upon receiving a GET request this endpoint will return a real quiote said by Nicolacus Maximus himself.",
    response_description="A Quote object that contains the quote said by Nicolacus Maximus and the date when the quote was said.",
    response_model=Quote,
)
def get_quote():
    return {
        "quote": "Life is short so eat it all.",
        "year": 1950,
    }
