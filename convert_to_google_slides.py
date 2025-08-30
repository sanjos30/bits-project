#!/usr/bin/env python3
"""
Convert Markdown Presentation to Google Slides Format
"""

import re
from datetime import datetime

def convert_markdown_to_google_slides():
    """Convert markdown presentation to Google Slides format"""
    
    print("🚀 Converting Markdown to Google Slides Format...")
    print("=" * 60)
    
    # Read the markdown file
    with open('M_Tech_Presentation_PPT.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into slides
    slides = content.split('---')
    
    google_slides_content = []
    google_slides_content.append("# Google Slides Conversion Guide")
    google_slides_content.append(f"## Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    google_slides_content.append("")
    google_slides_content.append("## 📋 Instructions:")
    google_slides_content.append("1. Open [Google Slides](https://slides.google.com)")
    google_slides_content.append("2. Create a new presentation")
    google_slides_content.append("3. Copy each slide content below")
    google_slides_content.append("4. Paste into Google Slides")
    google_slides_content.append("5. Format using Google Slides tools")
    google_slides_content.append("")
    google_slides_content.append("## 🎨 Formatting Tips:")
    google_slides_content.append("- Use **Title** style for main headings")
    google_slides_content.append("- Use **Subtitle** style for secondary headings")
    google_slides_content.append("- Use **Normal text** for body content")
    google_slides_content.append("- Recreate tables using Google Slides table feature")
    google_slides_content.append("- Use bullet points for lists")
    google_slides_content.append("")
    google_slides_content.append("=" * 60)
    google_slides_content.append("")
    
    slide_number = 1
    
    for slide in slides:
        slide = slide.strip()
        if not slide:
            continue
            
        # Extract slide title
        lines = slide.split('\n')
        title = ""
        content_lines = []
        
        for line in lines:
            if line.startswith('# ') and not title:
                title = line.replace('# ', '').strip()
            elif line.startswith('## ') and not title:
                title = line.replace('## ', '').strip()
            else:
                content_lines.append(line)
        
        if title:
            google_slides_content.append(f"## SLIDE {slide_number}: {title}")
            google_slides_content.append("")
            google_slides_content.append("### Content to copy:")
            google_slides_content.append("")
            
            # Clean up content
            slide_content = '\n'.join(content_lines).strip()
            
            # Remove markdown formatting that won't work in Google Slides
            slide_content = re.sub(r'\*\*(.*?)\*\*', r'\1', slide_content)  # Remove bold
            slide_content = re.sub(r'\*(.*?)\*', r'\1', slide_content)      # Remove italic
            slide_content = re.sub(r'`(.*?)`', r'\1', slide_content)        # Remove code
            
            # Handle tables
            if '|' in slide_content:
                google_slides_content.append("⚠️ **TABLE DETECTED** - Recreate this table in Google Slides:")
                google_slides_content.append("")
            
            google_slides_content.append(slide_content)
            google_slides_content.append("")
            google_slides_content.append("---")
            google_slides_content.append("")
            
            slide_number += 1
    
    # Write the conversion guide
    with open('Google_Slides_Conversion_Guide.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(google_slides_content))
    
    print("✅ Conversion guide created!")
    print("📄 File: Google_Slides_Conversion_Guide.md")
    print("")
    print("🎯 Next steps:")
    print("   1. Open Google_Slides_Conversion_Guide.md")
    print("   2. Follow the instructions")
    print("   3. Copy each slide content to Google Slides")
    print("   4. Format using Google Slides tools")
    print("")
    print("💡 Pro Tips:")
    print("   - Use Google Slides themes for professional look")
    print("   - Add transitions between slides")
    print("   - Use speaker notes for your script")
    print("   - Test presentation mode before your defense")

def create_quick_google_slides_template():
    """Create a quick template for Google Slides"""
    
    template_content = """
# 🚀 Quick Google Slides Setup

## Step 1: Open Google Slides
1. Go to [slides.google.com](https://slides.google.com)
2. Click "Blank" to create a new presentation
3. Rename it to "M.Tech Project Defense"

## Step 2: Choose a Theme
1. Click "Theme" in the toolbar
2. Choose a professional theme (recommended: "Simple Light" or "Modern")
3. This will give you consistent formatting

## Step 3: Set Up Slide Layouts
1. **Title Slide:** Use "Title" layout for slide 1
2. **Content Slides:** Use "Title and Content" layout for most slides
3. **Section Headers:** Use "Section Header" layout for major sections

## Step 4: Copy Content
1. Open `M_Tech_Presentation_PPT.md`
2. Copy each slide section (between `---` dividers)
3. Paste into Google Slides
4. Format using the toolbar

## Step 5: Formatting Guidelines

### Headers
- **Slide Titles:** Use "Title" style (large, bold)
- **Section Headers:** Use "Heading 1" style
- **Subsection Headers:** Use "Heading 2" style

### Content
- **Body Text:** Use "Normal text" style
- **Bullet Points:** Use Google Slides bullet feature
- **Tables:** Recreate using Insert > Table

### Visual Elements
- **Emojis:** Copy from markdown (they should work)
- **Diagrams:** Recreate using Google Slides shapes
- **Charts:** Use Google Slides chart feature for data

## Step 6: Add Speaker Notes
1. Click "View" > "Speaker notes"
2. Copy content from `Presentation_Script.md`
3. Add your speaking notes for each slide

## Step 7: Final Touches
1. **Transitions:** Add slide transitions
2. **Animations:** Add entrance animations for key points
3. **Colors:** Use consistent color scheme
4. **Fonts:** Stick to 2-3 fonts maximum

## 🎯 Recommended Google Slides Settings

### Theme: "Simple Light"
- Clean, professional look
- Good contrast for readability
- Works well with emojis

### Fonts:
- **Title:** Roboto (Google's default)
- **Body:** Arial or Roboto
- **Size:** 24pt for titles, 18pt for body

### Colors:
- **Primary:** Blue (#4285F4)
- **Secondary:** Gray (#5F6368)
- **Accent:** Green (#34A853) for success metrics

## 📱 Mobile-Friendly Tips
- Test on mobile device
- Ensure text is readable on small screens
- Keep bullet points concise
- Use large, clear fonts

## 🎤 Presentation Mode Tips
- Use "Present" button to test
- Check speaker notes visibility
- Test laser pointer feature
- Practice timing with each slide

---

**Ready to create your Google Slides presentation! 🚀**
"""
    
    with open('Google_Slides_Quick_Setup.md', 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    print("✅ Quick setup guide created!")
    print("📄 File: Google_Slides_Quick_Setup.md")

if __name__ == "__main__":
    print("🎯 Google Slides Conversion Tools")
    print("=" * 50)
    
    # Create conversion guide
    convert_markdown_to_google_slides()
    
    print("\n" + "=" * 50)
    
    # Create quick setup guide
    create_quick_google_slides_template()
    
    print("\n" + "=" * 50)
    print("🎉 Conversion tools ready!")
    print("\n📋 Files created:")
    print("   • Google_Slides_Conversion_Guide.md - Step-by-step conversion")
    print("   • Google_Slides_Quick_Setup.md - Quick setup guide")
    print("\n🚀 You're ready to create your Google Slides presentation!") 