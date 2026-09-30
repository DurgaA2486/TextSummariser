import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# Setup Ollama
OLLAMA_BASE_URL = "http://localhost:11434/v1"

ollama = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"
)

# Extract text from PDF
def extract_text_from_pdf(pdf_file):

    pdf_reader = PdfReader(pdf_file)

    text = ""

    for page in pdf_reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n\n"

    return text

def summarize_text(input,word_limit, file_name ="manual_input_summary.txt", translate=False, target_language=None):
    chunks = chunk_by_paragraphs(input)

    st.write(f"Total Chunks Created: {len(chunks)}")

    # Step 3 — Summarize chunks
    all_summaries = []

    progress = st.progress(0, text="Summarising starting !!!!")

    for i, chunk in enumerate(chunks):

        if len(chunks) <= 5:
            chunk_limit = max(150, word_limit // max(1, len(chunks)))
        elif len(chunks) <= 15:
            chunk_limit = max(120, word_limit // max(1, len(chunks)))
        else:
            chunk_limit = max(80, word_limit // max(1, len(chunks)))

        summary = get_summary(chunk, chunk_limit)

        all_summaries.append(summary)

        progress.progress(
            (i + 1) / len(chunks),
            text=f"Summarized chunk {i+1} of {len(chunks)}... {int(((i+1)/len(chunks))*100)}%"
        )

    # Step 4 — Combine summaries
    with st.spinner("Combining chunk summaries into final summary..."):
        final_summary = hierarchical_combine(
                all_summaries,
                word_limit
            )
    progress.progress(1.0, text="✅ Done! 100%")

    final_word_count = len(final_summary.split())
    st.caption(f"Final summary word count: {final_word_count} (target: {word_limit})")

    st.subheader("Final Summary")

    st.write(final_summary)

    # Save summary to file
    with open(
        file_name,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(final_summary)
    
    st.download_button(
        label="📥 Download Summary",
        data=final_summary,
        file_name="final_summary.txt",
        mime="text/plain"
    )

    if translate and target_language:
        with st.spinner(f"Translating to {target_language}..."):
            translated = translate_text(final_summary, target_language)
        
        st.subheader(f"Translated Summary ({target_language})")
        st.write(translated)
        
        st.download_button(
            label=f"📥 Download {target_language} Summary",
            data=translated,
            file_name=f"summary_{target_language.lower()}.txt",
            mime="text/plain"
        )


# Paragraph-based chunking
def chunk_by_paragraphs(text, max_words = 1500):

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:

        words = para.split()
        word_count = len(words)

        if current_word_count + word_count > max_words:

            chunks.append(" ".join(current_chunk))

            current_chunk = [para]
            current_word_count = word_count

        else:

            current_chunk.append(para)
            current_word_count += word_count

    if current_chunk:

        chunks.append(" ".join(current_chunk))

    return chunks


# Summarize chunk
def get_summary(text, word_limit = 150):

    prompt = f"""Summarize the following text in no fewer than {int(word_limit*0.85)} words and no more than {word_limit} words.

                STRICT INSTRUCTIONS:
                - Use ONLY information explicitly present in the given text.
                - Do NOT add, assume, or infer any events, characters, or details.
                - If a detail is unclear or missing, SKIP it.
                - Do NOT guess.

                CONTENT RULES:
                - Focus on major characters, major events, important locations, and key outcomes.
                - Preserve information that is important to understanding the overall story.
                - Do not omit major plot developments even if they seem repetitive.
                - Maintain chronological order.
                - Avoid minor details, repetition, and unnecessary descriptions.
                - Maintain correct sequence of events.
                - Keep the summary logically consistent.

                ACCURACY CHECK:
                - Do NOT change names, roles, dates, outcomes, or cause-and-effect relationships.
                - Do NOT merge or invent events.
                - Ensure every statement can be directly traced back to the input text.
                - If information is unclear or missing, omit it.
                - Do NOT infer motives, reasons, consequences, or relationships that are not explicitly stated.
                - Do NOT rewrite factual relationships in a different way unless they are clearly supported by the text.

                STYLE:
                - Write in clear, simple, and formal language.
                - Avoid dramatic or interpretive language.
                - Keep it factual, not descriptive storytelling.

                TEXT TO SUMMARIZE:
                {text}

                OUTPUT FORMAT:
                OUTPUT FORMAT:
                - Write the summary in a single coherent paragraph.
                - Return ONLY the summary.
                - Do NOT add introductions, explanations, titles, conclusions, or commentary.
                - Do NOT write phrases such as:
                "Here is the summary"
                "Here's a summary"
                "Final Summary"
                "In conclusion"
            """

    try:

        response = ollama.chat.completions.create(
            model="llama3.2:latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Error occurred: {str(e)}"
    
def combine_summaries(summaries, word_limit):
    combined = "\n\n".join(summaries)

    prompt = f"""
    The following are summaries of different sections of a book.

    Write a complete book summary that is between {int(word_limit*0.85)} and {word_limit} words. Use the full word budget by elaborating on details, sequences, and outcomes already present in the section summaries below — do not stop early just because you covered the main points.

    Requirements:
    - Cover the beginning, middle, and ending of the story.
    - Mention major characters.
    - Preserve the overall plot progression.
    - Do not focus disproportionately on the final chapters.
    - Include important events from throughout the book.
    - Maintain chronological order.
    - Do NOT stop before reaching the minimum word count — expand on stated details rather than ending early.
    - Return only the summary.
    IMPORTANT:
    - Include ONLY information present in the section summaries.
    - Do NOT add, infer, assume, or change facts.
    - Do NOT introduce new cause-and-effect relationships.
    - Prioritize thorough coverage over brevity for this task.
    - If unsure, omit the detail.
    - Every sentence must be supported by the provided section summaries.

    Sections:
    {combined}
    Before writing the final answer:
    - Verify that every sentence is supported by the provided summaries.
    - Remove any unsupported statement.
    - Remove any inferred relationship not explicitly stated.
    - Return ONLY the final summary paragraph.
    - Do NOT include introductions, explanations, headings, or commentary.
    """
    try:

        response = ollama.chat.completions.create(
            model="llama3.2:latest",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Error occurred: {str(e)}"


def expand_summary(current_summary, target_min, target_max, max_attempts=2):
    for attempt in range(max_attempts):
        word_count = len(current_summary.split())
        if word_count >= target_min:
            break

        prompt = f"""Here is a draft summary:

        {current_summary}

        This draft is only {word_count} words. Expand it to approximately {target_max} words by elaborating further on details, sequences, and outcomes that are already present in the draft above — describe them more fully rather than adding new facts.

        IMPORTANT:
        - Do NOT introduce any new facts, names, or events not already in the draft.
        - Do NOT repeat sentences verbatim; rewrite with more elaboration.
        - Return ONLY the expanded summary paragraph, nothing else.
        """

        try:
            response = ollama.chat.completions.create(
                model="llama3.2:latest",
                messages=[{"role": "user", "content": prompt}]
            )
            current_summary = response.choices[0].message.content
        except Exception as e:
            return current_summary  # bail out, return what we have

    return current_summary


def hierarchical_combine(summaries, word_limit):

    target_min = int(word_limit * 0.85)

    # Small inputs: combine directly
    if len(summaries) <= 15:
        final = combine_summaries(
            summaries,
            word_limit
        )
        return expand_summary(final, target_min, word_limit)

    # Large inputs: combine in stages
    group_size = 5

    section_summaries = []

    for i in range(
        0,
        len(summaries),
        group_size
    ):

        group = summaries[
            i:i+group_size
        ]

        section_summary = combine_summaries(
            group,
            min(
                300,
                max(
                    150,
                    word_limit // 3
                )
            )
        )

        section_summaries.append(
            section_summary
        )

    # If still too many sections,
    # combine again

    if len(section_summaries) > 10:

        second_level = []

        for i in range(
            0,
            len(section_summaries),
            group_size
        ):

            group = section_summaries[
                i:i+group_size
            ]

            summary = combine_summaries(
                group,
                300
            )

            second_level.append(
                summary
            )

        section_summaries = second_level

    final = combine_summaries(
        section_summaries,
        word_limit
    )
    return expand_summary(final, target_min, word_limit)

def translate_text(text, target_language):
    prompt = f"""
    Translate the following text to {target_language}.
    
    IMPORTANT:
    - Translate the COMPLETE text, every sentence.
    - Only return the translated text, nothing else.
    - Do NOT summarize or shorten.
    - Do NOT skip any part.
    - Do NOT add anything new.

    Text:
    {text}
    """
    try:
        response = ollama.chat.completions.create(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error occurred: {str(e)}"

# Streamlit UI
def main():

    st.title("📚 Textbook & Novel Summarizer (Ollama)")

    input_method = st.radio(
        "Choose your input method:",
        ("Upload PDF", "Enter Text Manually")
    )

    user_input = ""

    uploaded_file = None

    if input_method == "Upload PDF":

        uploaded_file = st.file_uploader(
            "Upload a PDF file",
            type=["pdf"]
        )

    else:

        user_input = st.text_area(
            "Enter text manually"
        )

    word_limit = st.number_input(
        "Summary word limit:",
        min_value=10,
        step=10,
        value=900
    )
    translate = st.checkbox("Translate Summary?")
    target_language = None

    if translate:
        target_language = st.selectbox(
            "Select language:",
            ["Spanish", "French", "German", 
            "Arabic", "Chinese", "Japanese", "Portuguese"]
        )

    if st.button("Generate Summary"):

        if uploaded_file:

            # Step 1 — Extract text
            text = extract_text_from_pdf(
                uploaded_file
            )

            summarize_text(text, word_limit, uploaded_file.name.replace(".pdf", "_summary.txt"), translate, target_language)

        elif user_input.strip():

            summarize_text(user_input, word_limit, translate = translate, target_language = target_language)

        else:

            st.warning(
                "Please upload PDF or enter text."
            )


if __name__ == "__main__":
    main()