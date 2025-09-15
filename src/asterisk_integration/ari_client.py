import asyncio
import logging
import asyncari # Changed from ari
from config import ARI_URL, ARI_USERNAME, ARI_PASSWORD, ARI_APP_NAME
from core.call_handler import AI_Call_Handler
from asyncari import exceptions # Import exceptions for specific handling

async def start_ari_client():
    """
    Connects to the Asterisk REST Interface (ARI) and starts listening for events.
    """
    logging.info("Attempting to connect to ARI...")
    
    while True:
        try:
            client = await asyncari.connect(ARI_URL, ARI_APP_NAME, ARI_USERNAME, ARI_PASSWORD) # Changed from ari.connect
            logging.info("ARI client connected successfully.")

            # Register event handlers (Changed from on_channel_event to on_event)
            client.on_event('StasisStart', on_stasis_start)
            client.on_event('StasisEnd', on_stasis_end)

            # Start the event loop
            await client.run()

        except exceptions.AsyncariException as e: # Changed exception type
            logging.error(f"Failed to connect to ARI: {e}. Retrying in 5 seconds...")
        except Exception as e:
            logging.critical(f"An unexpected error occurred with the ARI client: {e}", exc_info=True)
            logging.error("Retrying in 5 seconds...")
        
        await asyncio.sleep(5)

async def on_stasis_start(obj, ev):
    """
    Event handler for when a channel enters the Stasis application.
    This is the entry point for a new call.
    """
    channel = obj.get('channel')
    if not channel:
        logging.warning("StasisStart event received without a channel object.")
        return

    logging.info(f"New call (StasisStart) on channel: {channel.id}")
    
    # Create a handler for the call and run it as a background task
    handler = AI_Call_Handler(channel)
    asyncio.create_task(handler.handle_call())

async def on_stasis_end(obj, ev):
    """
    Event handler for when a channel leaves the Stasis application.
    """
    channel = obj.get('channel')
    if channel:
        logging.info(f"Call ended (StasisEnd) on channel: {channel.id}")
        # Future cleanup logic for the call can be added here.