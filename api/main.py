from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import base64
import os
import io
from PIL import Image
import openai
from openai import OpenAI
from dotenv import load_dotenv
import re
import json

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")

# Validate API key
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable is not set!")
    print("Please set your OpenAI API key in the .env file")
    # We'll continue but log the error

print(f"API key loaded: {'Yes' if api_key else 'No'}")
print(f"API key length: {len(api_key) if api_key else 0}")

# Initialize OpenAI client
client = OpenAI(
    api_key=api_key
)

app = FastAPI(title="Food Recognition API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Food item model
class FoodItem(BaseModel):
    name: str
    calories: float
    protein: float
    carbs: float
    fat: float
    confidence: float
    image: Optional[str] = None
    selected: bool = True

# Response model
class ImageRequest(BaseModel):
    image_data: str

class FoodRecognitionResponse(BaseModel):
    items: List[FoodItem]
    message: Optional[str] = None

# Nutrition database (fallback when API doesn't provide complete nutrition info)
nutrition_database = {
    "apple": {"calories": 95, "protein": 0.5, "carbs": 25, "fat": 0.3},
    "banana": {"calories": 105, "protein": 1.3, "carbs": 27, "fat": 0.4},
    "chicken breast": {"calories": 165, "protein": 31, "carbs": 0, "fat": 3.6},
    "rice": {"calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3},
    "broccoli": {"calories": 55, "protein": 3.7, "carbs": 11.2, "fat": 0.6},
    "salmon": {"calories": 206, "protein": 22, "carbs": 0, "fat": 13},
    "yogurt": {"calories": 100, "protein": 10, "carbs": 4, "fat": 5},
    "avocado": {"calories": 160, "protein": 2, "carbs": 8.5, "fat": 14.7},
    "oatmeal": {"calories": 150, "protein": 5, "carbs": 27, "fat": 2.5},
    "eggs": {"calories": 78, "protein": 6.3, "carbs": 0.6, "fat": 5.3},
}

def get_nutrition_info(food_name: str) -> dict:
    """Get nutrition information for a food item from our database"""
    food_name_lower = food_name.lower()
    
    # Try exact match first
    if food_name_lower in nutrition_database:
        return nutrition_database[food_name_lower]
    
    # Try partial match
    for key in nutrition_database:
        if key in food_name_lower or food_name_lower in key:
            return nutrition_database[key]
    
    # Default values if no match found
    return {
        "calories": 100,
        "protein": 2,
        "carbs": 15,
        "fat": 2
    }

@app.get("/")
async def root():
    return {"message": "Food Recognition API is running"}

@app.get("/nutrition")
async def get_nutrition_info(food: str):
    """Get nutritional information for a specific food item"""
    try:
        # Call OpenAI API to get nutrition information
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a nutrition expert. Provide accurate nutritional information for food items."
                },
                {
                    "role": "user",
                    "content": f"Provide nutritional information for {food}. Return only a JSON object with these fields: name, calories, protein (g), carbs (g), and fat (g). Be precise and accurate."
                }
            ],
            max_tokens=150
        )
        
        # Process the response
        ai_response = response.choices[0].message.content
        
        # Extract JSON from the response
        json_match = re.search(r'{.*}', ai_response, re.DOTALL)
        if json_match:
            try:
                nutrition_data = json.loads(json_match.group(0))
                return nutrition_data
            except json.JSONDecodeError:
                # Fallback to database if JSON parsing fails
                return get_nutrition_info(food)
        else:
            # Fallback to database if no JSON found
            return get_nutrition_info(food)
    
    except Exception as e:
        # Fallback to database on any error
        return get_nutrition_info(food)

@app.post("/analyze/base64", response_model=FoodRecognitionResponse)
async def analyze_base64_image(request: ImageRequest):
    """
    Analyze a base64 encoded image to identify food items
    """
    try:
        # Get the image data from the request
        image_data = request.image_data
        
        # Log for debugging
        print(f"Received image data of length: {len(image_data) if image_data else 'None'}")
        print(f"Data starts with: {image_data[:50] if image_data else 'None'}")
        
        # Check if we received a URL instead of base64 data
        if image_data and (image_data.startswith('http://') or image_data.startswith('https://')):
            raise ValueError("Received URL instead of base64 image data. Please provide actual image data.")
        
        # Ensure we have valid image data
        if not image_data:
            raise ValueError("No image data provided. Please capture or upload an image first.")
        
        # Handle data URLs (remove prefix if present)
        if image_data and image_data.startswith('data:'):
            print("Detected data URL format, extracting base64 content...")
            try:
                # Extract the base64 content from data URL
                image_data = image_data.split(',')[1]
                print(f"Extracted base64 content, new length: {len(image_data)}")
            except IndexError:
                raise ValueError("Invalid data URL format. Could not extract base64 content.")
        
        # Validate that the image data looks like base64
        if not re.match(r'^[A-Za-z0-9+/]+={0,2}$', image_data):
            # Print a sample of the data for debugging
            print(f"Invalid base64 format detected. Sample: {image_data[:30]}...")
            print(f"Characters in sample: {[ord(c) for c in image_data[:20]]}")
            raise ValueError("Invalid base64 format. Please provide correctly formatted base64 image data.")
        
        try:
            # Prepare the base64 image for OpenAI API
            base64_image_with_prefix = f"data:image/jpeg;base64,{image_data}"
            
            print("Sending request to OpenAI API...")
            print(f"Base64 image prefix length: {len(base64_image_with_prefix) if base64_image_with_prefix else 'None'}")
            
            # Add additional error handling and debugging
            try:
                # Call OpenAI API
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Identify all food items in this image. For each item, provide: name, estimated calories, protein (g), carbs (g), and fat (g). Format your response as a JSON array with these fields for each food item. Be specific about the food items."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": base64_image_with_prefix
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=800
                )
                print("OpenAI API request successful")
            except Exception as api_error:
                print(f"OpenAI API error: {type(api_error).__name__}")
                print(f"Error message: {str(api_error)}")
                print(f"Error details: {repr(api_error)}")
                raise api_error
            
            # Process the response
            ai_response = response.choices[0].message.content
            print(f"OpenAI response: {ai_response[:100]}...")
            
            # Initialize food_items list
            food_items = []
            
            # Check if the response indicates no food was found
            if "no food" in ai_response.lower() or "no food items" in ai_response.lower():
                # Return an empty list of food items with a message
                return FoodRecognitionResponse(
                    items=[],
                    message="No food items were detected in the image. Please try a different image with visible food items."
                )
            
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
            if json_match:
                try:
                    food_items_data = json.loads(json_match.group(0))
                    # Process food items
                    for item in food_items_data:
                        name = item.get("name", "Unknown Food")
                        
                        # Get nutrition data
                        nutrition = {
                            "calories": item.get("calories"),
                            "protein": item.get("protein"),
                            "carbs": item.get("carbs"),
                            "fat": item.get("fat")
                        }
                        
                        # Fill in missing data
                        if any(v is None for v in nutrition.values()):
                            db_nutrition = get_nutrition_info(name)
                            for key, value in nutrition.items():
                                if value is None:
                                    nutrition[key] = db_nutrition[key]
                        
                        # Create food item
                        food_item = FoodItem(
                            name=name,
                            calories=float(nutrition["calories"]),
                            protein=float(nutrition["protein"]),
                            carbs=float(nutrition["carbs"]),
                            fat=float(nutrition["fat"]),
                            confidence=90.0,  # Default confidence
                            image=f"https://placehold.co/100x100/cccccc/black?text={name.replace(' ', '+')}",
                            selected=True
                        )
                        food_items.append(food_item)
                    
                    # Return the response with food items
                    return FoodRecognitionResponse(items=food_items)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    # Provide detailed error for JSON decode issues
                    error_message = str(e)
                    if not error_message or error_message.strip() == "":
                        error_message = "Invalid JSON format in OpenAI response"
                    
                    raise HTTPException(
                        status_code=422,
                        detail=f"Failed to parse OpenAI response as JSON: {error_message}. The AI response was: {ai_response[:200]}"
                    )
            else:
                print("No JSON found in OpenAI response")
                # Check if the response indicates no food was found
                if "no food" in ai_response.lower() or "no food items" in ai_response.lower():
                    # Return an empty list of food items with a message
                    return FoodRecognitionResponse(
                        items=[],
                        message="No food items were detected in the image. Please try a different image with visible food items."
                    )
                else:
                    # Provide detailed error for missing JSON
                    raise HTTPException(
                        status_code=422,
                        detail=f"No valid JSON array found in OpenAI response. The AI response was: {ai_response[:200]}"
                    )
        except Exception as e:
            print(f"Error processing image: {e}")
            # Provide detailed error with exception information
            error_type = type(e).__name__
            error_message = str(e)
            
            # Check if the error is empty and if the AI response indicates no food
            if (not error_message or error_message.strip() == "") and 'ai_response' in locals():
                if "no food" in ai_response.lower() or "no food items" in ai_response.lower():
                    # Return an empty list with a helpful message instead of an error
                    return FoodRecognitionResponse(
                        items=[],
                        message="No food items were detected in the image. Please try a different image with visible food items."
                    )
                else:
                    error_message = f"Unknown {error_type} error occurred during image processing"
            
            # If we reach here, it's a genuine error
            raise HTTPException(
                status_code=500,
                detail=f"Error processing image: {error_type} - {error_message}"
            )
        
        # If we reach here, it means we didn't find any JSON in the response
        return FoodRecognitionResponse(
            items=[],
            message="No food items were detected in the image. Please try a different image with visible food items."
        )
    
    except Exception as e:
        print(f"Error in analyze_base64_image: {str(e)}")
        # Return a response with an error message instead of raising an exception
        return FoodRecognitionResponse(
            items=[],
            message=f"Error analyzing image: {str(e)}"
        )

@app.post("/analyze/upload", response_model=FoodRecognitionResponse)
async def analyze_uploaded_image(file: UploadFile = File(...)):
    """
    Analyze an uploaded image file to identify food items
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Convert to base64
        base64_image = base64.b64encode(contents).decode("utf-8")
        
        # Prepare the base64 image for OpenAI API
        base64_image_with_prefix = f"data:image/jpeg;base64,{base64_image}"
        
        try:
            # Call OpenAI API directly (no coroutines)
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Identify all food items in this image. For each item, provide: name, estimated calories, protein (g), carbs (g), and fat (g). Format your response as a JSON array with these fields for each food item. Be specific about the food items."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": base64_image_with_prefix
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800
            )
            
            # Process the response
            ai_response = response.choices[0].message.content
            print(f"OpenAI response: {ai_response[:100]}...")
            
            # Initialize food_items list
            food_items = []
            
            # Check if the response indicates no food was found
            if "no food" in ai_response.lower() or "no food items" in ai_response.lower():
                return FoodRecognitionResponse(
                    items=[],
                    message="No food items were detected in the image. Please try a different image with visible food items."
                )
            
            # Extract JSON from the response
            json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
            if not json_match:
                return FoodRecognitionResponse(
                    items=[],
                    message="Could not recognize food items in the image. Please try a clearer image of food."
                )
            
            # Parse the JSON data
            try:
                food_items_data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return FoodRecognitionResponse(
                    items=[],
                    message="Could not parse food data from the image. Please try a clearer image of food items."
                )
            
            # Process each food item
            for item in food_items_data:
                try:
                    name = item.get("name", "Unknown Food")
                    
                    # Get nutrition data
                    nutrition = {
                        "calories": item.get("calories", 0),
                        "protein": item.get("protein", 0),
                        "carbs": item.get("carbs", 0),
                        "fat": item.get("fat", 0)
                    }
                    
                    # Fill in missing data
                    if any(v == 0 for v in nutrition.values()):
                        # Use the non-async version of get_nutrition_info to avoid coroutine issues
                        db_nutrition = {
                            "calories": 100,
                            "protein": 2,
                            "carbs": 15,
                            "fat": 2
                        }
                        
                        # Try to find the food in our local database
                        food_name_lower = name.lower()
                        
                        # Try exact match first
                        if food_name_lower in nutrition_database:
                            db_nutrition = nutrition_database[food_name_lower]
                        else:
                            # Try partial match
                            for key in nutrition_database:
                                if key in food_name_lower or food_name_lower in key:
                                    db_nutrition = nutrition_database[key]
                                    break
                        
                        # Fill in missing values
                        for key, value in nutrition.items():
                            if value == 0:
                                nutrition[key] = db_nutrition[key]
                    
                    # Create food item
                    food_item = FoodItem(
                        name=name,
                        calories=float(nutrition["calories"]),
                        protein=float(nutrition["protein"]),
                        carbs=float(nutrition["carbs"]),
                        fat=float(nutrition["fat"]),
                        confidence=90.0,  # Default confidence
                        image=f"https://placehold.co/100x100/cccccc/black?text={name.replace(' ', '+')}",
                        selected=True
                    )
                    food_items.append(food_item)
                except Exception as item_error:
                    print(f"Error processing food item: {item_error}")
                    # Continue with other items
                    continue
            
            # Return the response with food items
            if food_items:
                return FoodRecognitionResponse(items=food_items)
            else:
                return FoodRecognitionResponse(
                    items=[],
                    message="Could not process any food items from the image. Please try a clearer image."
                )
                
        except Exception as api_error:
            print(f"OpenAI API error: {api_error}")
            return FoodRecognitionResponse(
                items=[],
                message=f"Error analyzing image: {str(api_error)}"
            )
    
    except Exception as e:
        print(f"Upload error: {e}")
        return FoodRecognitionResponse(
            items=[],
            message=f"Error processing uploaded image: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
