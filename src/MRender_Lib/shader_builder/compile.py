import os
import re
import subprocess

temp_dir_name = '.temp'

include_pattern = re.compile(r'^\s*#include\s+"(.+)"')
version_pattern = re.compile(r'^\s*#version\b')
in_decl_pattern = re.compile(r'^\s*(in)\s+(.*?;)\s*$')
out_decl_pattern = re.compile(r'^\s*(out)\s+(.*?;)\s*$')

def scan_shader_files(input_dir, shader_extensions):
	shader_sources = {}

	temp_dir = os.path.join(os.path.abspath(input_dir), temp_dir_name)
	for root, _, files in os.walk(input_dir):
		abs_root = os.path.abspath(root)
		if abs_root.startswith(temp_dir):
			continue

		for file in files:
			ext = os.path.splitext(file)[1]
			if ext in shader_extensions:
				full_path = os.path.join(root, file)
				with open(full_path, 'r', encoding='utf-8') as f:
					shader_sources[os.path.abspath(full_path)] = f.read()

	return shader_sources

def preprocess_shader(path, shader_sources, included_files=None):
	if included_files is None:
		included_files = set()

	normalized_path = os.path.abspath(path)
	if normalized_path in included_files:
		return ""

	included_files.add(normalized_path)

	source = shader_sources.get(normalized_path)
	if source is None:
		raise FileNotFoundError(f"Shader file not found in sources: {path}")

	result = []
	base_dir = os.path.dirname(normalized_path)

	for line in source.splitlines(keepends=True):
		match = include_pattern.match(line)
		if match:
			include_rel_path = match.group(1)
			include_full_path = os.path.abspath(os.path.join(base_dir, include_rel_path))
			included_content = preprocess_shader(include_full_path, shader_sources, included_files)
			result.append(f"// Begin include: {include_rel_path}\n")
			result.append(included_content)
			result.append(f"// End include: {include_rel_path}\n")
		else:
			result.append(line)

	return ''.join(result)

def add_layout_locations(shader_source):
	lines = shader_source.splitlines(keepends=True)
	result = []
	location_counter_in = 0
	location_counter_out = 0

	has_version = False
	for line in lines:
		if version_pattern.match(line):
			if has_version is False:
				has_version = True
				result.append(line)
			continue

		in_match = in_decl_pattern.match(line)
		out_match = out_decl_pattern.match(line)

		if in_match:
			new_line = f"layout(location = {location_counter_in}) in {in_match.group(2)}\n"
			result.append(new_line)
			location_counter_in += 1
		elif out_match:
			new_line = f"layout(location = {location_counter_out}) out {out_match.group(2)}\n"
			result.append(new_line)
			location_counter_out += 1
		else:
			result.append(line)

	return ''.join(result)

def compile_shader(preprocessed_path, output_spv_path):
	stage = None
	ext = os.path.splitext(preprocessed_path)[1]
	if ext == '.vert':
		stage = 'vertex'
	elif ext == '.frag':
		stage = 'fragment'
	elif ext == '.comp':
		stage = 'compute'
	elif ext == '.geom':
		stage = 'geometry'
	elif ext == '.tesc':
		stage = 'tesscontrol'
	elif ext == '.tese':
		stage = 'tesseval'
	elif ext == '.glsl':
		return
	else:
		return
	
	result = subprocess.run(
		["glslc", "-fshader-stage=" + stage, preprocessed_path, "-o", output_spv_path],
		capture_output=True, text=True
	)

	if result.returncode != 0:
		print(f"[ERROR] {output_spv_path} compile failed:\n{result.stderr}")
		return
	else:
		print(f"[OK] {output_spv_path}")
		return
	
def file_needs_rebuild(source_path, target_path):
	if not os.path.exists(target_path):
		return True
	return os.path.getmtime(source_path) > os.path.getmtime(target_path)

def compile_all_shaders(input_dir, output_dir):
	temp_dir = os.path.join(input_dir, '.temp')

	shader_extensions = {'.vert', '.frag', '.glsl'}
	print("Scanning shaders...")
	shader_sources = scan_shader_files(input_dir, shader_extensions)

	temp_dir = os.path.join(os.path.abspath(input_dir), temp_dir_name)
	for root, _, files in os.walk(input_dir):
		abs_root = os.path.abspath(root)
		if abs_root.startswith(temp_dir):
			continue

		for file in files:
			ext = os.path.splitext(file)[1]
			if ext in shader_extensions:
				input_path = os.path.join(root, file)

				# 상대 경로
				rel_path = os.path.relpath(input_path, input_dir)

				# temp 폴더에 변환된 glsl 저장
				preprocessed_path = os.path.join(temp_dir, rel_path)
				os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)

				# spv 출력 경로
				output_path = os.path.join(output_dir, rel_path + ".spv")
				os.makedirs(os.path.dirname(output_path), exist_ok=True)

				# 변환
				try:
					preprocessed_code = preprocess_shader(input_path, shader_sources)
					preprocessed_code = add_layout_locations(preprocessed_code)
				except Exception as e:
					print(f"[ERROR] Preprocess failed for {input_path}: {e}")
					continue

				# 이전 temp 파일이랑 비교
				rebuild = True
				if os.path.exists(preprocessed_path):
					with open(preprocessed_path, 'r', encoding='utf-8') as f:
						old_content = f.read()
					if old_content == preprocessed_code:
						rebuild = False

				if rebuild:
					with open(preprocessed_path, 'w', encoding='utf-8') as f:
						f.write(preprocessed_code)
					print(f"[Update] {preprocessed_path}")

				# temp가 수정됐거나 spv가 없거나 오래됐으면 재컴파일
				if rebuild or file_needs_rebuild(preprocessed_path, output_path):
					compile_shader(preprocessed_path, output_path)

if __name__ == "__main__":
	import sys

	if len(sys.argv) != 3:
		print("Usage: python compile_all_shaders.py <input_shader_folder> <temp_folder> <output_spv_folder>")
		sys.exit(1)

	current_file_path = os.path.abspath(__file__)
	os.chdir(os.path.expanduser(os.path.dirname(current_file_path)))

	input_shader_dir = sys.argv[1]
	output_spv_dir = sys.argv[2]

	compile_all_shaders(input_shader_dir, output_spv_dir)