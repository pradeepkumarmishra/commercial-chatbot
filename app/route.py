from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.routers import SemanticRouter

# Your existing routes with minor fixes
faq_route = Route(
    name="faq",
    utterances=[
        # Return policy variations
        "What is the return policy?",
        "How can I return a product?",
        "What is your refund policy?",
        "Can I return items?",

        # Payment and discounts
        "Do I get discount with HDFC credit card?",
        "What payment methods do you accept?",
        "Can I use my credit card?",
        "Are there any bank offers?",

        # Order tracking and management
        "How can I track my order?",
        "Where is my order?",
        "Can I cancel my order?",
        "How to modify my order?",

        # Promotions and codes
        "How do I use a promo code?",
        "How to apply discount code?",
        "Are there any ongoing sales?",
        "What are current offers?",

        # Shipping and delivery
        "Do you offer international shipping?",
        "What are shipping charges?",
        "How long does delivery take?",

        # Issues and support
        "What if I receive damaged product?",
        "How to report a problem?",
        "Need help with my order",
    ]
)

product_route = Route(
    name="product",
    utterances=[
        # Product search and browsing
        "Show me shoes under 200 INR",
        "I want to buy shoes",
        "Looking for affordable footwear",
        "Show me cheap shoes",

        # Category browsing
        "Show me trending clothes for men",
        "What's new in men's fashion?",
        "Latest men's clothing",
        "Trendy outfits for men",

        # General product queries
        "What products do you have?",
        "Show me your catalog",
        "I want to browse products",
        "What's available?",

        # Specific product requests
        "Show me latest collection",
        "New product launches",
        "What's recently launched?",
        "Show me new arrivals",

        # Discount-related product queries
        "Products on 50% discount",
        "Discounted items",
        "Sale products",
        "Clearance items",
    ]
)

rts = [faq_route, product_route]
encoder = HuggingFaceEncoder(name="sentence-transformers/all-mpnet-base-v2")
router = SemanticRouter(encoder = encoder, routes = rts, auto_sync="local")

if __name__ == "__main__":
    query = "track my order"
    selected_route =  router(query)
    # Use .score() to get similarity scores for all routes
    print(selected_route.name)

