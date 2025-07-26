"""
A script to automate the building of the portfolio-analytics-ai package.

This script cleans previous build artifacts and then creates both a source
distribution (sdist) and a binary wheel distribution (wheel).

Usage:
    python build_package.py
"""
import os
import sys
import shutil
import subprocess

# Get the project root directory (the parent of the directory containing this script)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def clean():
    """
    Removes old build artifacts.
    """
    print("Cleaning previous build artifacts...")
    
    # Directories to remove
    dirs_to_remove = [
        os.path.join(PROJECT_ROOT, 'dist'),
        os.path.join(PROJECT_ROOT, 'build')
    ]
    
    # Find and remove .egg-info directories
    for root, dirs, _ in os.walk(os.path.join(PROJECT_ROOT, 'src')):
        for d in dirs:
            if d.endswith('.egg-info'):
                dirs_to_remove.append(os.path.join(root, d))

    for dir_path in dirs_to_remove:
        if os.path.isdir(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")
            except OSError as e:
                print(f"Error removing directory {dir_path}: {e}", file=sys.stderr)
                sys.exit(1)

def build():
    """
    Builds the source and wheel distributions.
    """
    print("\nBuilding source and wheel distributions...")
    
    # Ensure the 'build' package is installed
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'build'])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install 'build' package: {e}", file=sys.stderr)
        sys.exit(1)

    # Command to build the package
    command = [sys.executable, '-m', 'build']
    
    try:
        # Run the build command from the project root
        process = subprocess.Popen(
            command, 
            cwd=PROJECT_ROOT, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code != 0:
            print(f"\nBuild failed with return code {return_code}", file=sys.stderr)
            sys.exit(return_code)
            
    except FileNotFoundError:
        print("Error: 'python' or 'pip' command not found.", file=sys.stderr)
        print("Please ensure Python is installed and in your system's PATH.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during the build: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """
    Main function to run the clean and build process.
    """
    # Change to the project root directory to ensure commands run correctly
    os.chdir(PROJECT_ROOT)
    
    print("=============================================")
    print("  Building Portfolio Analytics AI Package  ")
    print("=============================================")
    
    # 1. Clean previous builds
    clean()
    
    # 2. Build the package
    build()
    
    print("\n=============================================")
    print("  Build process completed successfully!      ")
    print("=============================================")
    print(f"Distribution files are located in: {os.path.join(PROJECT_ROOT, 'dist')}")
    
    # List the created files
    dist_dir = os.path.join(PROJECT_ROOT, 'dist')
    if os.path.isdir(dist_dir):
        print("\nGenerated files:")
        for filename in os.listdir(dist_dir):
            print(f"- {filename}")

if __name__ == "__main__":
    main()
