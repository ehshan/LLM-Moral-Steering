import pandas as pd
import time

def process_csv_in_chunks(file_path, row_processor_func, chunk_size=100000, max_results=None):
    """
    A general-purpose function to process a large CSV file in chunks.

    Args:
        file_path (str or pathlib.Path): Path to the large CSV file.
        row_processor_func (function): A function to call on each row.
                                       This function should accept a row (pd.Series)
                                       and return a result (or None to skip).
        chunk_size (int): Number of rows to read into memory at a time.
        max_results (int, optional): If provided, stops processing after
                                     this many results have been collected.

    Returns:
        list: A list containing all the non-None results returned by
              the row_processor_func.
        int: Total number of rows scanned in the file.
        int: Total number of rows skipped (where processor returned None).
    """
    
    print(f"[+] Starting chunked processing for: {file_path}")
    print(f"    Chunk size: {chunk_size}")
    if max_results:
        print(f"    Stopping after: {max_results} results")

    start_time = time.time()
    results_list = []
    total_rows_scanned = 0
    
    try:
        # Create the CSV iterator
        csv_iterator = pd.read_csv(
            file_path, 
            chunksize=chunk_size, 
            low_memory=True
        )

        # Loop through each chunk
        for i, chunk_df in enumerate(csv_iterator):
            print(f"    Processing chunk {i+1} (Rows {total_rows_scanned + 1} - {total_rows_scanned + len(chunk_df)})...")
            
            # --- Perform common cleaning on the CHUNK ---
            # Can add any universal cleaning steps here if needed later
            # For now row_processor_func will handle everything
            # as cleaning might be specific to the script (e.g., CHARACTER_COLS)
            
            # Iterate through rows in the current chunk
            for _, row in chunk_df.iterrows():
                
                # Apply the user-provided function to the row
                result = row_processor_func(row)
                
                if result is not None:
                    # If the function returned something, save it
                    results_list.append(result)

                # Check if we've hit our target number of results
                if max_results and len(results_list) >= max_results:
                    break # Stop processing rows
            
            total_rows_scanned += len(chunk_df)

            if max_results and len(results_list) >= max_results:
                print(f"\n    Target of {max_results} results reached. Stopping chunk processing.")
                break # Stop processing new chunks
        
        end_time = time.time()
        print(f"\n[+] Chunked processing complete.")
        print(f"    Total time taken: {end_time - start_time:.2f} seconds.")
        print(f"    Total rows scanned: {total_rows_scanned}")
        print(f"    Total results collected: {len(results_list)}")
        
        # Calculate skipped rows based on results vs. scanned
        # Note: This is an approximation if max_results is hit
        skipped_rows = total_rows_scanned - len(results_list)
        return results_list, total_rows_scanned, skipped_rows

    except FileNotFoundError:
        print(f"ERROR: File not found at '{file_path}'.")
        return [], 0, 0
    except Exception as e:
        print(f"ERROR: An error occurred while processing chunks: {e}")
        return [], total_rows_scanned, 0